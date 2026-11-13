from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tabulus import cli
from tabulus.evaluation import evaluate_step7_physical
from tabulus.resolved_export import ENRICHMENT_COLUMNS


@pytest.fixture
def physical_pair(tmp_path):
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "prediction"
    gold_dir.mkdir()
    pred_dir.mkdir()
    columns = list(ENRICHMENT_COLUMNS)
    gold_columns = [c for c in columns if c not in {
        "tabulus_resolution_statuses", "tabulus_resolution_sources",
        "tabulus_resolution_confidences", "tabulus_resolution_reasons", "tabulus_unmatched_tokens",
    }]

    def write_csv(path, metadata, value="ZrCl4"):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerows([
                ["Material", "Precursor"] + metadata,
                ["ZrO2", value] + ["[]"] * len(metadata),
            ])

    write_csv(gold_dir / "a.csv", gold_columns)
    write_csv(gold_dir / "b.csv", gold_columns)
    write_csv(pred_dir / "p.csv", columns, " ZrCl4 ")
    gold = {
        "schema_version": 1, "quality_columns": gold_columns,
        "tables": [
            {"table_key": key, "table_id": i, "gold_csv": filename, "header_row_index": 0}
            for i, (key, filename) in enumerate((("fragment-a", "a.csv"), ("fragment-b", "b.csv")), 1)
        ],
    }
    pred = {
        "schema_version": 1, "enrichment_columns": columns,
        "tables": [{"table_id": 20, "resolved_csv": "p.csv", "header_row_index": 0}],
    }
    alignment = {
        "schema_version": 1,
        "tables": [
            {"table_id": 20, "table_key": "fragment-a"},
            {"table_id": 21, "table_key": "fragment-b"},
        ],
    }
    paths = {
        "gold": gold_dir / "gold.json",
        "prediction": pred_dir / "resolved_tables.json",
        "alignment": tmp_path / "alignment.json",
    }
    for name, value in (("gold", gold), ("prediction", pred), ("alignment", alignment)):
        paths[name].write_text(json.dumps(value), encoding="utf-8")
    return paths


def test_cli_scores_both_metrics_and_penalizes_missing_export(physical_pair, tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", [
        "tabulus", "evaluate-step7-physical",
        "--prediction", str(physical_pair["prediction"]),
        "--gold", str(physical_pair["gold"]),
        "--alignment", str(physical_pair["alignment"]),
        "--out", str(output),
    ])
    cli.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["coverage"] == {"expected": 2, "exported": 1, "missing": 1, "empty": 0}
    assert result["macro"] == {
        metric: {"precision": 50, "recall": 50, "f1": 50} for metric in ("exact", "rms")
    }
    assert result["exact_grid_matches"] == 1
    assert result["tables"][0]["exact_cells"]["gold_nonempty_cells"] == 4
    assert result["tables"][0]["prediction_table_id"] == 20
    assert all(item["sha256"] for item in result["inputs"])


def test_mapping_is_explicit_and_one_to_one(physical_pair):
    with pytest.raises(ValueError, match="explicit mapping"):
        evaluate_step7_physical(physical_pair["prediction"], physical_pair["gold"], metric="exact")
    alignment = json.loads(physical_pair["alignment"].read_text())
    alignment["tables"].append(alignment["tables"][0])
    physical_pair["alignment"].write_text(json.dumps(alignment))
    with pytest.raises(ValueError, match="duplicate alignment"):
        evaluate_step7_physical(
            physical_pair["prediction"], physical_pair["gold"],
            alignment_path=physical_pair["alignment"], metric="exact",
        )


@pytest.mark.parametrize("missing", [False, True])
def test_empty_and_absent_csvs_receive_zero(physical_pair, missing):
    prediction_csv = physical_pair["prediction"].parent / "p.csv"
    if missing:
        prediction_csv.unlink()
    else:
        prediction_csv.write_text("")
    result = evaluate_step7_physical(
        physical_pair["prediction"], physical_pair["gold"],
        alignment_path=physical_pair["alignment"],
    )
    assert result["macro"]["rms"]["f1"] == result["macro"]["exact"]["f1"] == 0
    assert result["coverage"]["exported"] == 0
    assert result["coverage"]["empty"] == int(not missing)
    assert result["coverage"]["missing"] == 1 + int(missing)


def test_exact_is_sensitive_to_scientific_transcription_and_metadata_header_is_validated(physical_pair):
    prediction_csv = physical_pair["prediction"].parent / "p.csv"
    prediction_csv.write_text(prediction_csv.read_text(encoding="utf-8-sig").replace("ZrCl4", "ZrCly"), encoding="utf-8")
    result = evaluate_step7_physical(
        physical_pair["prediction"], physical_pair["gold"],
        alignment_path=physical_pair["alignment"],
    )
    assert result["tables"][0]["metrics"]["exact"]["recall"] == 75
    assert 75 < result["tables"][0]["metrics"]["rms"]["f1"] < 100
    prediction_csv.write_text(prediction_csv.read_text().replace("tabulus_reference_indices", "wrong_header"))
    with pytest.raises(ValueError, match="metadata"):
        evaluate_step7_physical(
            physical_pair["prediction"], physical_pair["gold"],
            alignment_path=physical_pair["alignment"], metric="exact",
        )


def test_cli_prevents_overwriting_an_input(physical_pair, monkeypatch):
    path = physical_pair["prediction"]
    original = path.read_bytes()
    monkeypatch.setattr("sys.argv", [
        "tabulus", "evaluate-step7-physical", "--metric", "exact",
        "--prediction", str(path), "--gold", str(physical_pair["gold"]),
        "--alignment", str(physical_pair["alignment"]), "--out", str(path),
    ])
    with pytest.raises(ValueError, match="aliases"):
        cli.main()
    assert path.read_bytes() == original
