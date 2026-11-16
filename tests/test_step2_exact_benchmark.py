from __future__ import annotations

import csv
import json
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def saved_benchmark(tmp_path):
    root = tmp_path / "benchmark"
    paper = root / "domain/topic/P1"
    metadata = root / "metadata/tables.csv"
    metadata.parent.mkdir(parents=True)
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "paper_id", "table_key", "table_id", "page_nr", "gold_table_path",
        ])
        writer.writeheader()
        for tid in (1, 2):
            path = paper / f"step2_table_reconstruction/tables/fragment-{tid}/gold.csv"
            path.parent.mkdir(parents=True)
            path.write_text("row,value\na,10\n")
            writer.writerow({
                "paper_id": "P1", "table_key": f"fragment-{tid}", "table_id": tid,
                "page_nr": tid, "gold_table_path": path.relative_to(root).as_posix(),
            })
    pred = paper / "tabulus_runs/step2/controlled/run_01/model/reconstruction/predictions"
    pred.mkdir(parents=True)
    (pred / "table_001.csv").write_text("row,value\na,10\n")
    module = runpy.run_path(str(
        Path(__file__).parents[1] / "scripts/evaluation/step2/evaluate_exact_match.py"
    ))
    return root, paper, pred, module["score_benchmark"]


def test_benchmark_counts_missing_predictions_and_checks_population(saved_benchmark, tmp_path):
    root, _, _, score = saved_benchmark
    with pytest.raises(ValueError, match="Expected 610"):
        score(root, tmp_path / "wrong", adapters=["model"])
    result = score(root, tmp_path / "scores", adapters=["model"], expected_fragments=2)
    report = result["methods"]["model"]
    assert report["coverage"] == {"scored": 1, "missing": 1}
    assert report["macro"] == {"precision": 50, "recall": 50, "f1": 50}
    assert result["inference_rerun"] is False
    with pytest.raises(ValueError, match="must be empty"):
        score(root, tmp_path / "scores", adapters=["model"], expected_fragments=2)


def test_benchmark_unusable_prediction_has_zero_precision(saved_benchmark, tmp_path):
    root, _, pred, score = saved_benchmark
    (pred / "table_002.csv").write_text("")
    result = score(root, tmp_path / "scores", adapters=["model"], expected_fragments=2)
    assert result["methods"]["model"]["coverage"] == {"scored": 1, "unusable": 1}
    assert result["methods"]["model"]["macro"]["precision"] == 50


def test_native_reuses_saved_csv_and_parses_added_saved_body(saved_benchmark, tmp_path):
    root, paper, _, score = saved_benchmark
    native = tmp_path / "native"
    native.mkdir()
    (native / "prediction.csv").write_text("row,value\na,10\n")
    (native / "table_results.csv").write_text(
        "paper,table_id,status,prediction_csv\nP1,1,ok,prediction.csv\n"
    )
    index = paper / "tabulus_runs/step1/production/artifacts/table-crops/tables_index.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"tables": [{
        "table_id": 20, "page_nr": 2, "image_name": "fragment-2.png",
        "mineru_table_body": "<table><tr><td>row</td><td>value</td></tr><tr><td>a</td><td>10</td></tr></table>",
    }]}))
    result = score(
        root, tmp_path / "scores", adapters=["model"], expected_fragments=2, native_root=native,
    )
    assert result["methods"]["mineru-native"]["coverage"] == {"scored": 2}
    assert result["methods"]["mineru-native"]["macro"]["f1"] == 100
    assert (tmp_path / "scores/native_predictions/P1/table_002.csv").is_file()
    changed = json.loads(index.read_text())
    changed["tables"][0]["table_id"] = 2
    changed["tables"][0]["image_name"] = "unrelated-fragment.png"
    index.write_text(json.dumps(changed))
    unmatched = score(
        root, tmp_path / "unmatched-native", adapters=["model"],
        expected_fragments=2, native_root=native,
    )
    assert unmatched["methods"]["mineru-native"]["coverage"] == {"scored": 1, "missing": 1}


def test_controlled_production_and_step7_share_canonical_gold(saved_benchmark, tmp_path):
    root, paper, _, score = saved_benchmark
    from tabulus.resolved_export import ENRICHMENT_COLUMNS
    production = paper / "tabulus_runs/step2/production/run_01/model/reconstruction/predictions/table_020.csv"
    production.parent.mkdir(parents=True)
    production.write_text("row,value\na,10\n")
    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({"schema_version": 1, "tables": [
        {"paper_id": "P1", "table_key": "fragment-1", "prediction_table_id": 20},
    ]}))
    metrics = ("rms", "exact-mapping", "exact")
    controlled = score(root, tmp_path / "controlled", adapters=["model"], expected_fragments=2,
                       metrics=metrics, include_native=False)
    produced = score(root, tmp_path / "production", adapters=["model"], expected_fragments=2,
                     metrics=metrics, include_native=False, setting="production", alignment_path=alignment)
    assert controlled["groups"]["controlled/model"] == produced["groups"]["production/model"]
    resolved = tmp_path / "resolved.csv"
    with resolved.open("w", newline="") as handle:
        csv.writer(handle).writerows([
            ["row", "value", *ENRICHMENT_COLUMNS], ["a", "10", *(["[]"] * len(ENRICHMENT_COLUMNS))],
        ])
    manifest = tmp_path / "resolved_tables.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "enrichment_columns": list(ENRICHMENT_COLUMNS),
        "tables": [{"table_id": 20, "resolved_csv": str(resolved), "header_row_index": 0}],
    }))
    mapping = json.loads(alignment.read_text())
    mapping["tables"][0]["prediction_manifest"] = str(manifest)
    alignment.write_text(json.dumps(mapping))
    exported = score(root, tmp_path / "step7", adapters=["model"], expected_fragments=2,
                     metrics=metrics, include_native=False, setting="step7", alignment_path=alignment)
    assert exported["groups"]["step7/model"] == controlled["groups"]["controlled/model"]
    mapping["tables"].append({
        "paper_id": "P1", "table_key": "fragment-2", "prediction_table_id": 20,
        "prediction_manifest": str(manifest),
    })
    alignment.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="multiple gold"):
        score(root, tmp_path / "bad", adapters=["model"], expected_fragments=2,
              setting="step7", alignment_path=alignment)


def test_subset_population_share_and_resume(saved_benchmark, tmp_path):
    root, _, _, score = saved_benchmark
    fragments = tmp_path / "fragments.json"
    fragments.write_text(json.dumps({"fragments": [{"paper_id": "P1", "table_key": "fragment-1"}]}))
    kwargs = dict(adapters=["model"], expected_fragments=2, fragments=fragments)
    initial = score(root, tmp_path / "subset", **kwargs)
    assert initial["selected_population"]["fragments"] == 1
    assert initial["full_population"]["fragments"] == 2
    assert all(value == .5 for value in initial["population_share"].values())
    assert score(root, tmp_path / "subset", resume=True, **kwargs) == initial



def test_partial_archive_is_only_a_verification_baseline(saved_benchmark, tmp_path):
    root, paper, pred, score = saved_benchmark
    (pred / "table_002.csv").write_text("row,value\na,10\n")
    archive = root / "results/step2/controlled/run_01/model/rms/per-table.json"
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps([{
        "paper_id": "P1", "table_id": 1, "status": "ok",
        "gold_csv": "/old/server/P1/reference_tables/tables/fragment-1/gold.csv",
        "precision": 100, "recall": 100, "f1": 100,
    }]))
    result = score(root, tmp_path / "partial", adapters=["model"], expected_fragments=2, metrics=("rms",))
    report = result["methods"]["model"]
    assert report["coverage"] == {"scored": 2}
    assert report["macro"]["f1"] == 100
    verified = report["archive_verification"]
    assert verified["archived_fragments"] == verified["checked"] == 1
    assert verified["without_baseline"] == [{"paper_id": "P1", "table_key": "fragment-2"}]
    assert verified["mismatches"] == []
    historical = json.loads(archive.read_text())
    historical[0]["f1"] = 99
    archive.write_text(json.dumps(historical))
    result = score(root, tmp_path / "mismatch", adapters=["model"], expected_fragments=2, metrics=("rms",))
    verified = result["methods"]["model"]["archive_verification"]
    assert verified["checked"] == 1
    assert verified["mismatches"][0]["archived"]["f1"] == 99
    assert result["methods"]["model"]["macro"]["f1"] == 100


def test_archive_status_does_not_override_actual_prediction_inputs(saved_benchmark, tmp_path):
    root, _, pred, score = saved_benchmark
    (pred / "table_002.csv").write_text("row,value\na,10\n")
    archive = root / "results/step2/controlled/run_01/model/rms/per-table.json"
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps([{
        "paper_id": "P1", "table_id": 1, "status": "empty",
        "gold_csv": "/historical/P1/reference_tables/tables/fragment-1/gold.csv",
        "precision": 0, "recall": 0, "f1": 0,
    }]))
    result = score(root, tmp_path / "actual", adapters=["model"], expected_fragments=2, metrics=("rms",))
    report = result["methods"]["model"]
    assert report["coverage"] == {"scored": 2}
    assert report["macro"]["f1"] == 100
    assert report["archive_verification"]["mismatches"][0]["recalculated"]["f1"] == 100


def test_windows_archive_paths_use_stable_fragment_keys(saved_benchmark, tmp_path):
    root, _, pred, score = saved_benchmark
    (pred / "table_002.csv").write_text("row,value\na,10\n")
    archive = root / "results/step2/controlled/run_01/model/rms/per-table.json"
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps([{
        "paper_id": "P1", "table_id": 1, "status": "ok",
        "gold_csv": r"C:\historical\P1\reference_tables\tables\fragment-1\gold.csv",
        "precision": 100, "recall": 100, "f1": 100,
    }]))
    result = score(root, tmp_path / "windows", adapters=["model"], expected_fragments=2, metrics=("rms",))
    assert result["methods"]["model"]["archive_verification"]["checked"] == 1
