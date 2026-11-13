from __future__ import annotations

import json

import pytest

from tabulus import cli
from tabulus.evaluation import evaluate_table_reconstruction, exact_mapping_match


TABLE = "material | temperature | pressure\nal2o3 | 300 | 2\ntio2 | 250 | 1"


@pytest.mark.parametrize(("prediction", "expected"), [
    (TABLE, (1, 1, 1)),
    ("material | temperature | pressure\nal2o3 | 301 | 2\ntio2 | 250 | 1", (.75, .75, .75)),
    ("material | temperature | pressure\nal2o3 | 300.0 | 2\ntio2 | 250 | 1", (.75, .75, .75)),
    ("material | temperature | pressure\nal2o3 | 300 | 2", (1, .5, 2 / 3)),
    (TABLE + "\nzno | 200 | 3", (2 / 3, 1, .8)),
    ("material | temperature | pressure\nal203 | 300 | 2\ntio2 | 250 | 1", (.5, .5, .5)),
    ("material | pressure | temperature\ntio2 | 1 | 250\nal2o3 | 2 | 300", (1, 1, 1)),
    ("material | al2o3 | tio2\ntemperature | 300 | 250\npressure | 2 | 1", (1, 1, 1)),
])
def test_exact_mapping_requires_equal_keys_and_string_values(prediction, expected):
    result = exact_mapping_match(TABLE, prediction)
    assert (result.precision, result.recall, result.f1) == pytest.approx(expected)


@pytest.mark.parametrize(("gold", "prediction", "expected"), [
    ("", "", (1, 1, 1)),
    (TABLE, "", (1, 0, 0)),
    ("", TABLE, (0, 1, 0)),
    ("row | value", "", (1, 1, 1)),
    ("title | example", "title | Example", (1, 1, 1)),
    ("row | value\na | ", "row | value\na | ", (1, 1, 1)),
])
def test_exact_mapping_empty_inputs_titles_and_blank_values_follow_rms(gold, prediction, expected):
    result = exact_mapping_match(gold, prediction)
    assert (result.precision, result.recall, result.f1) == pytest.approx(expected)


def test_repeated_keys_use_the_same_last_value_as_rms():
    result = exact_mapping_match("row | value\na | 1\na | 2", "row | value\na | 2")
    assert result.f1 == 1


def test_public_api_reuses_csv_normalization_and_preserves_artifact_semantics(tmp_path):
    gold = tmp_path / "gold.csv"
    prediction = tmp_path / "prediction.csv"
    gold.write_text("\ufeffMaterial, Deposition   temperature ,Notes\nAl2O3,300,\n", encoding="utf-8")
    prediction.write_text("material,Deposition temperature,notes\nAL2O3, 300 ,\n", encoding="utf-8")
    result = evaluate_table_reconstruction(gold, prediction, metric="exact-mapping")
    assert result.precision == result.recall == result.f1 == 100
    payload = result.to_dict()
    assert payload["metric"] == "exact-mapping"
    assert "exact_cells" not in payload
    assert "text_threshold" not in payload
    legacy = evaluate_table_reconstruction(gold, prediction, metric="exact")
    assert legacy.to_dict()["metric"] == "exact"
    assert "exact_cells" in legacy.to_dict()
    rms = evaluate_table_reconstruction(gold, prediction)
    assert rms.to_dict()["metric"] == "rms"
    assert rms.to_dict()["text_threshold"] == .5
    assert rms.to_dict()["number_threshold"] == .1


def test_cli_dispatches_exact_mapping_and_writes_report(tmp_path, monkeypatch, capsys):
    gold, prediction, output = (tmp_path / name for name in ("gold.csv", "prediction.csv", "report.json"))
    gold.write_text("row,value\na,10\nb,20\n")
    prediction.write_text("row,value\na,10.0\nb,20\n")
    monkeypatch.setattr("sys.argv", [
        "tabulus", "evaluate-table-reconstruction", "--metric", "exact-mapping",
        "--gold", str(gold), "--prediction", str(prediction), "--out", str(output),
    ])
    cli.main()
    assert json.loads(output.read_text())["f1"] == 50
    assert "Exact Mapping Match" in capsys.readouterr().out
