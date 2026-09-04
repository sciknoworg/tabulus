from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabulus.evaluation import (
    evaluate_table_reconstruction,
    relative_mapping_similarity,
)
from tabulus.evaluation.rms import csv_to_rms_text


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_rms_exact_match_is_one() -> None:
    table = "year | argentina | brazil\n1999 | 200 | 158"

    scores = relative_mapping_similarity(table, table)

    assert scores.precision == pytest.approx(1.0)
    assert scores.recall == pytest.approx(1.0)
    assert scores.f1 == pytest.approx(1.0)


def test_rms_matches_deplot_reference_example() -> None:
    target = (
        "title | my table\n"
        "year | argentina | brazil\n"
        "1999 | 200 | 158"
    )
    prediction = (
        "title | my table\n"
        "year | argentina | brazil\n"
        "1999 | 202 | 0"
    )

    scores = relative_mapping_similarity(target, prediction)

    expected = 1.99 / 3.0
    assert scores.precision == pytest.approx(expected)
    assert scores.recall == pytest.approx(expected)
    assert scores.f1 == pytest.approx(expected)


def test_rms_is_invariant_to_row_and_column_order() -> None:
    target = (
        "material | temperature | pressure\n"
        "Al2O3 | 300 | 2\n"
        "TiO2 | 250 | 1"
    )
    prediction = (
        "material | pressure | temperature\n"
        "TiO2 | 1 | 250\n"
        "Al2O3 | 2 | 300"
    )

    scores = relative_mapping_similarity(target, prediction)

    assert scores.f1 == pytest.approx(1.0)


def test_rms_accepts_transposed_prediction() -> None:
    target = (
        "material | temperature | pressure\n"
        "Al2O3 | 300 | 2\n"
        "TiO2 | 250 | 1"
    )
    prediction = (
        "material | Al2O3 | TiO2\n"
        "temperature | 300 | 250\n"
        "pressure | 2 | 1"
    )

    scores = relative_mapping_similarity(target, prediction)

    assert scores.f1 == pytest.approx(1.0)


def test_csv_evaluation_returns_rms_provenance_on_deplot_scale(
    tmp_path: Path,
) -> None:
    gold = _write_csv(
        tmp_path / "gold.csv",
        "material,temperature\nAl2O3,300\n",
    )
    prediction = _write_csv(
        tmp_path / "prediction.csv",
        "material,temperature\nAl2O3,300\n",
    )

    result = evaluate_table_reconstruction(gold, prediction)

    assert result.metric == "rms"
    assert result.metric_name == "Relative Mapping Similarity"
    assert result.metric_short_name == "RMS"
    assert result.implementation == "DePlot"
    assert result.score_scale == "[0,100]"
    assert result.precision == pytest.approx(100.0)
    assert result.recall == pytest.approx(100.0)
    assert result.f1 == pytest.approx(100.0)


def test_evaluation_json_is_written_only_when_explicitly_requested(
    tmp_path: Path,
) -> None:
    gold = _write_csv(tmp_path / "gold.csv", "row,value\na,1\n")
    prediction = _write_csv(tmp_path / "prediction.csv", "row,value\na,1\n")
    output = tmp_path / "evaluation" / "result.json"

    result = evaluate_table_reconstruction(gold, prediction)

    assert not output.exists()
    written = result.write_json(output)
    assert written == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metric_name"] == "Relative Mapping Similarity"
    assert payload["f1"] == pytest.approx(100.0)


def test_csv_conversion_handles_bom_and_ragged_rows(tmp_path: Path) -> None:
    table = _write_csv(
        tmp_path / "table.csv",
        "\ufeffrow,value,notes\na,1\nb,2,x\n",
    )

    converted = csv_to_rms_text(table)

    assert converted == "row | value | notes\na | 1 | \nb | 2 | x"


def test_missing_prediction_is_an_error(tmp_path: Path) -> None:
    gold = _write_csv(tmp_path / "gold.csv", "row,value\na,1\n")

    with pytest.raises(FileNotFoundError, match="Prediction CSV not found"):
        evaluate_table_reconstruction(gold, tmp_path / "missing.csv")


def test_rms_thresholds_must_be_probabilities() -> None:
    with pytest.raises(ValueError, match="text_threshold"):
        relative_mapping_similarity("a | b", "a | b", text_threshold=1.1)

    with pytest.raises(ValueError, match="number_threshold"):
        relative_mapping_similarity("a | b", "a | b", number_threshold=-0.1)
