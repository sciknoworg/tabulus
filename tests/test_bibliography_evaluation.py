from __future__ import annotations

import json
from pathlib import Path

import pytest

import tabulus.evaluation.bibliography as bibliography_evaluation

from tabulus.evaluation import evaluate_bibliography


def write_json(path: Path, obj) -> Path:
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_identical_bibliography_is_perfect(
    tmp_path: Path,
) -> None:
    raw = (
        "T. Suntola, Mater. Sci. Rep. "
        "4, 261 (1989)."
    )

    gold = write_json(
        tmp_path / "gold.json",
        [
            {
                "nr": "1",
                "ref": raw,
            }
        ],
    )

    prediction = write_json(
        tmp_path / "bibliography.json",
        {
            "bibliography_count": 1,
            "bibliography_source": "grobid",
            "entries": [
                {
                    "index": 1,
                    "raw": raw,
                    "doi": "",
                    "source": "grobid",
                    "title": "",
                    "authors": ["T Suntola"],
                    "year": 1989,
                    "venue": "Mater. Sci. Rep",
                    "volume": "4",
                    "issue": "",
                    "pages": "261",
                }
            ],
        },
    )

    result = evaluate_bibliography(
        gold,
        prediction,
    )

    assert result.gold_entries == 1
    assert result.predicted_entries == 1

    assert result.content_recovered_gold == 1
    assert result.content_supported_predictions == 1

    assert result.content_precision == 1.0
    assert result.content_recall == 1.0
    assert result.content_f1 == 1.0

    assert result.strict_entry_true_positives == 1
    assert result.strict_entry_false_positives == 0
    assert result.strict_entry_false_negatives == 0
    assert result.strict_entry_precision == 1.0
    assert result.strict_entry_recall == 1.0
    assert result.strict_entry_f1 == 1.0

    payload = result.to_dict()
    assert payload["strict_entry_true_positives"] == 1
    assert payload["strict_entry_false_positives"] == 0
    assert payload["strict_entry_false_negatives"] == 0
    assert payload["strict_entry_precision"] == 1.0
    assert payload["strict_entry_recall"] == 1.0
    assert payload["strict_entry_f1"] == 1.0

    assert result.one_to_one_gold == 1
    assert result.split_gold_groups == 0
    assert result.merge_groups == 0

    assert result.unmatched_gold == 0
    assert result.unmatched_pred == 0

    assert result.position_preserved_gold == 1
    assert result.position_preservation_rate == 1.0

    assert len(result.operations) == 1
    assert result.operations[0]["kind"] == "one_to_one"


def test_prediction_count_must_match_entries(
    tmp_path: Path,
) -> None:
    gold = write_json(
        tmp_path / "gold.json",
        [
            {
                "nr": "1",
                "ref": "Reference one.",
            }
        ],
    )

    prediction = write_json(
        tmp_path / "bibliography.json",
        {
            "bibliography_count": 2,
            "entries": [
                {
                    "index": 1,
                    "raw": "Reference one.",
                }
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="bibliography_count",
    ):
        evaluate_bibliography(
            gold,
            prediction,
        )


def test_gold_requires_reference_string(
    tmp_path: Path,
) -> None:
    gold = write_json(
        tmp_path / "gold.json",
        [{"nr": "1"}],
    )

    prediction = write_json(
        tmp_path / "bibliography.json",
        {
            "bibliography_count": 0,
            "entries": [],
        },
    )

    with pytest.raises(
        ValueError,
        match="string 'ref'",
    ):
        evaluate_bibliography(
            gold,
            prediction,
        )



def test_strict_entry_metrics_penalize_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gold = write_json(
        tmp_path / "gold.json",
        [
            {
                "nr": "1",
                "ref": "A. Author, First reference, 2020.",
            },
            {
                "nr": "2",
                "ref": "B. Author, Second reference, 2021.",
            },
        ],
    )

    prediction = write_json(
        tmp_path / "bibliography.json",
        {
            "bibliography_count": 3,
            "entries": [
                {
                    "index": 1,
                    "raw": "A. Author, First reference, 2020.",
                },
                {
                    "index": 2,
                    "raw": "B. Author, Second",
                },
                {
                    "index": 3,
                    "raw": "reference, 2021.",
                },
            ],
        },
    )

    def fake_align_paper(
        gold_entries,
        pred_entries,
    ):
        assert len(gold_entries) == 2
        assert len(pred_entries) == 3

        return (
            [
                {
                    "kind": "one_to_one",
                    "gold_start": 1,
                    "gold_count": 1,
                    "pred_start": 1,
                    "pred_count": 1,
                    "score": 1.0,
                },
                {
                    "kind": "split",
                    "gold_start": 2,
                    "gold_count": 1,
                    "pred_start": 2,
                    "pred_count": 2,
                    "score": 1.0,
                },
            ],
            [],
            [],
        )

    monkeypatch.setattr(
        bibliography_evaluation,
        "align_paper",
        fake_align_paper,
    )

    result = (
        bibliography_evaluation.evaluate_bibliography(
            gold,
            prediction,
        )
    )

    # Content recovery is perfect because the split content
    # is fully accounted for.
    assert result.content_precision == 1.0
    assert result.content_recall == 1.0
    assert result.content_f1 == 1.0

    # Strict entry recovery counts only the 1:1 operation.
    assert result.strict_entry_true_positives == 1
    assert result.strict_entry_false_positives == 2
    assert result.strict_entry_false_negatives == 1

    assert result.strict_entry_precision == pytest.approx(
        1 / 3
    )
    assert result.strict_entry_recall == pytest.approx(
        1 / 2
    )
    assert result.strict_entry_f1 == pytest.approx(
        0.4
    )

    payload = result.to_dict()

    assert (
        payload["strict_entry_true_positives"]
        == 1
    )
    assert (
        payload["strict_entry_false_positives"]
        == 2
    )
    assert (
        payload["strict_entry_false_negatives"]
        == 1
    )
