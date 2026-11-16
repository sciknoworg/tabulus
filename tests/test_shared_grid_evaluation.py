from dataclasses import replace
import csv
import json

import pytest

from tabulus.evaluation import (
    FragmentPair, GridInput, aggregate_grid_results, evaluate_pairs,
    evaluate_step7_physical, evaluate_table_reconstruction, score_table_grids,
)
from tabulus.evaluation.batch import aggregate_evaluation_runs
from tabulus.resolved_export import ENRICHMENT_COLUMNS


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path


def make_pairs(tmp_path):
    return [
        FragmentPair(str(i),
                     GridInput(write(tmp_path / f"g{i}.csv", [["r", "v"], [str(i), "10"]])),
                     GridInput(write(tmp_path / f"p{i}.csv", [["r", "v"], [str(i), "10"]])),
                     group=str(i % 2))
        for i in range(3)
    ]


def test_macro_micro_sizes_missing_and_no_cross_table_matching():
    small = score_table_grids([["r", "v"], ["a", "1"]], [["r", "v"], ["a", "1"]])
    gold = [["r", "v"], ["b", "2"], ["c", "3"], ["d", "4"]]
    large = score_table_grids(gold, [["r", "v"], ["b", "20"], ["c", "30"], ["d", "40"]])
    summary = aggregate_grid_results([small, large])
    assert summary["macro"]["exact-mapping"]["f1"] == 50
    assert summary["micro"]["exact-mapping"]["f1"] == 25
    assert summary["pooled_counts"]["rms"] == {
        "gold_count": 4, "predicted_count": 4, "matched_credit": 1, "unit": "mappings",
    }
    missing = score_table_grids(gold, [], status="missing")
    summary = aggregate_grid_results([small, missing])
    assert summary["micro"]["exact-mapping"] == {"precision": 100, "recall": 25, "f1": 40}
    unrelated = score_table_grids([], gold)
    assert aggregate_grid_results([missing, unrelated])["pooled_counts"]["exact-mapping"]["matched_credit"] == 0


def test_metric_distinctions_duplicate_counts_and_independent_orientations():
    row = score_table_grids([["r", "v"], ["a", "chloride"]], [["r", "v"], ["a", "chlorite"]])
    assert 0 < row["metrics"]["rms"]["f1"] < 100
    assert row["metrics"]["exact-mapping"]["f1"] == 0
    assert row["metrics"]["exact"]["f1"] == 75
    duplicate = score_table_grids([["r", "v"], ["a", "1"], ["a", "2"]], [["r", "v"], ["a", "2"]])
    assert duplicate["statistics"]["exact-mapping"]["gold_count"] == 1
    assert duplicate["statistics"]["exact"]["gold_count"] == 6
    assert duplicate["metrics"]["exact-mapping"]["f1"] == 100
    assert duplicate["metrics"]["exact"]["f1"] == 60
    row = score_table_grids(
        [["r", "a", "b"], ["x", "foo", "bar"], ["y", "baz", "qux"]],
        [["r", "a", "b"], ["x", "fo0", "bar0"], ["y", "baz0", "qux0"]],
    )
    assert row["statistics"]["rms"]["orientation"] == "original"
    assert row["statistics"]["exact-mapping"]["orientation"] == "transposed"


@pytest.mark.parametrize(("gold", "pred", "expected"), [
    ([], [], {"precision": 100, "recall": 100, "f1": 100}),
    ([["r", "v"], ["a", "1"]], [], {"precision": 100, "recall": 0, "f1": 0}),
    ([], [["r", "v"], ["a", "1"]], {"precision": 0, "recall": 100, "f1": 0}),
])
def test_zero_denominators(gold, pred, expected):
    row = score_table_grids(gold, pred)
    summary = aggregate_grid_results([row])
    assert summary["micro"]["rms"] == summary["micro"]["exact-mapping"] == expected
    assert score_table_grids([], [])["metrics"]["exact"]["f1"] == 0
    row = score_table_grids([], [], status="missing")
    assert row["metrics"]["rms"]["f1"] == 0
    assert aggregate_grid_results([row])["micro"]["rms"]["f1"] == 100


def test_serial_parallel_interrupt_and_resume(tmp_path):
    pairs = make_pairs(tmp_path)
    serial = evaluate_pairs(pairs, output_dir=tmp_path / "serial")
    parallel = evaluate_pairs(pairs, workers=2, output_dir=tmp_path / "parallel")
    assert serial == parallel
    assert (tmp_path / "serial/fragments.csv").read_bytes() == (tmp_path / "parallel/fragments.csv").read_bytes()
    def stop(_):
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_pairs(pairs, output_dir=tmp_path / "resume", on_result=stop)
    assert len(list((tmp_path / "resume/fragments").glob("*.json"))) == 1
    assert evaluate_pairs(pairs, output_dir=tmp_path / "resume", resume=True, workers=2) == serial


@pytest.mark.parametrize("change", ["input", "settings", "correspondence", "implementation"])
def test_stale_resume_rejected(tmp_path, monkeypatch, change):
    from tabulus.evaluation import batch
    pairs = make_pairs(tmp_path)
    out = tmp_path / "run"
    evaluate_pairs(pairs, metrics=("exact-mapping",), output_dir=out)
    metrics = ("exact-mapping",)
    if change == "input":
        pairs[0].prediction.path.write_text("r,v\na,changed\n")
    elif change == "settings":
        metrics = ("exact",)
    elif change == "correspondence":
        pairs[0] = replace(pairs[0], provenance={"table_key": "different"})
    else:
        monkeypatch.setattr(batch, "implementation_provenance", lambda: {"version": "different"})
    with pytest.raises(ValueError, match="Stale checkpoint"):
        evaluate_pairs(pairs, output_dir=out, resume=True, metrics=metrics)


def test_array_aggregation(tmp_path):
    pairs = make_pairs(tmp_path)
    for i, pair in enumerate(pairs):
        evaluate_pairs([pair], output_dir=tmp_path / f"task{i}")
    merged = aggregate_evaluation_runs([tmp_path / f"task{i}" for i in range(3)], tmp_path / "merged")
    assert merged["groups"] == evaluate_pairs(pairs)["groups"]
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_evaluation_runs([tmp_path / "task0", tmp_path / "task0"], tmp_path / "bad")


def test_canonical_step2_and_step7_have_identical_scores_and_counts(tmp_path):
    rows = [["tabulus_reference_indices", "value"], ["a", "10"]]
    root = tmp_path / "step2"
    gold = write(root / "tables/fragment-a/gold.csv", rows)
    (root / "tables_index.json").write_text(json.dumps({
        "tables": [{"table_id": 1, "benchmark_table_key": "fragment-a"}],
    }))
    raw = write(tmp_path / "raw.csv", rows)
    columns = list(ENRICHMENT_COLUMNS)
    resolved = write(tmp_path / "resolved.csv", [rows[0] + columns, rows[1] + ["[]"] * len(columns)])
    manifest = tmp_path / "resolved_tables.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "enrichment_columns": columns,
        "tables": [{"table_id": 20, "resolved_csv": str(resolved), "header_row_index": 0}],
    }))
    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({
        "schema_version": 1, "tables": [{"table_id": 20, "table_key": "fragment-a"}],
    }))
    result = evaluate_step7_physical(
        manifest, canonical_gold_root=root, alignment_path=alignment,
        metrics=("rms", "exact-mapping", "exact"),
    )
    for metric in ("rms", "exact-mapping", "exact"):
        expected = evaluate_table_reconstruction(gold, raw, metric=metric)
        assert result["tables"][0]["metrics"][metric] == {
            field: getattr(expected, field) for field in ("precision", "recall", "f1")
        }
        assert result["tables"][0]["statistics"][metric] == expected.statistics
    assert result["excluded_columns"]["gold"] == []
    assert result["population"]["gold_nonempty_cells"] == 4
    assert set(evaluate_step7_physical(
        manifest, canonical_gold_root=root, alignment_path=alignment, metric="both",
    )["macro"]) == {"exact", "rms"}


def test_batch_cli_and_metric_selection(tmp_path, monkeypatch):
    from tabulus import cli
    pair = make_pairs(tmp_path)[0]
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"schema_version": 1, "pairs": [{
        "fragment_id": "P1/fragment-a", "group": "controlled/model",
        "gold": {"csv": pair.gold.path.name}, "prediction": {"csv": pair.prediction.path.name},
    }]}))
    out = tmp_path / "cli-run"
    monkeypatch.setattr("sys.argv", [
        "tabulus", "evaluate-table-batch", "--manifest", str(manifest),
        "--out", str(out), "--metrics", "rms", "exact-mapping", "exact",
    ])
    cli.main()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["groups"]["controlled/model"]["micro"]["exact-mapping"]["f1"] == 100
    args = cli.build_parser().parse_args([
        "evaluate-step7-physical", "--prediction", "resolved_tables.json",
        "--step2-gold", "step2", "--metrics", "rms", "exact-mapping", "exact",
    ])
    assert args.metrics == ["rms", "exact-mapping", "exact"]
    assert args.metric == "both"


def test_checkpoint_integrity_and_single_writer_lock(tmp_path):
    pairs = make_pairs(tmp_path)
    out = tmp_path / "locked"
    def check_lock(_):
        with pytest.raises(ValueError, match="already in use"):
            evaluate_pairs(pairs, output_dir=out, resume=True)
    evaluate_pairs(pairs, output_dir=out, on_result=check_lock)
    checkpoint = next((out / "fragments").glob("*.json"))
    checkpoint.write_text("{}")
    with pytest.raises(ValueError, match="Stale or corrupted"):
        evaluate_pairs(pairs, output_dir=out, resume=True)



def test_unavailable_grid_equality_and_normalized_input_sequences():
    source = GridInput(inline_grid=[["r", "v"], ["a", "1"]], metadata_columns=[])
    assert hash(source)
    result = score_table_grids([], [], status="missing")
    assert result["exact_cells"]["exact_grid"] is False


def test_aggregation_rejects_changed_preparation_version_and_config_corruption(tmp_path):
    a, b, _ = make_pairs(tmp_path)
    a = replace(a, provenance={"benchmark_preparation_sha256": "version-a"})
    b = replace(b, provenance={"benchmark_preparation_sha256": "version-b"})
    evaluate_pairs([a], output_dir=tmp_path / "a")
    evaluate_pairs([b], output_dir=tmp_path / "b")
    with pytest.raises(ValueError, match="different"):
        aggregate_evaluation_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "bad")
    config = tmp_path / "a/run.json"
    data = json.loads(config.read_text())
    data["settings"]["number_threshold"] = .2
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Stale"):
        evaluate_pairs([a], output_dir=tmp_path / "a", resume=True)


def test_input_change_during_preparation_never_publishes_a_score(tmp_path, monkeypatch):
    from tabulus.evaluation import batch
    pairs = make_pairs(tmp_path)
    original = batch.load_grid
    def changing(source):
        grid = original(source)
        if source.path == pairs[0].prediction.path:
            source.path.write_text("r,v\nchanged,20\n")
        return grid
    monkeypatch.setattr(batch, "load_grid", changing)
    with pytest.raises(ValueError, match="changed during preparation"):
        evaluate_pairs(pairs, output_dir=tmp_path / "changing")
    assert not list((tmp_path / "changing/fragments").glob("*.json"))
