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
        "table_id": 2, "page_nr": 2,
        "mineru_table_body": "<table><tr><td>row</td><td>value</td></tr><tr><td>a</td><td>10</td></tr></table>",
    }]}))
    result = score(
        root, tmp_path / "scores", adapters=["model"], expected_fragments=2, native_root=native,
    )
    assert result["methods"]["mineru-native"]["coverage"] == {"scored": 2}
    assert result["methods"]["mineru-native"]["macro"]["f1"] == 100
    assert (tmp_path / "scores/native_predictions/P1/table_002.csv").is_file()
