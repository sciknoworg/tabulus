from __future__ import annotations

from pathlib import Path

import tabulus.cli as cli


def test_evaluate_bibliography_parser_accepts_pair() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "evaluate-bibliography",
            "--gold",
            "gold.json",
            "--prediction",
            "bibliography.json",
            "--out",
            "evaluation.json",
        ]
    )

    assert args.command == "evaluate-bibliography"
    assert args.gold == Path("gold.json")
    assert args.prediction == Path("bibliography.json")
    assert args.out == Path("evaluation.json")


def test_evaluate_bibliography_main_dispatches(
    monkeypatch,
) -> None:
    calls = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "evaluate-bibliography",
            "--gold",
            "gold.json",
            "--prediction",
            "bibliography.json",
            "--out",
            "evaluation.json",
        ],
    )

    class FakeResult:
        gold_path = Path("gold.json")
        prediction_path = Path("bibliography.json")

        gold_entries = 10
        predicted_entries = 11

        content_precision = 0.9
        content_recall = 1.0
        content_f1 = 0.947368421

        strict_entry_true_positives = 8
        strict_entry_false_positives = 3
        strict_entry_false_negatives = 2
        strict_entry_precision = 8 / 11
        strict_entry_recall = 8 / 10
        strict_entry_f1 = 16 / 21

        one_to_one_gold = 9
        split_gold_groups = 1
        merge_groups = 0

        unmatched_gold = 0
        unmatched_pred = 1

        def write_json(self, path):
            calls["out"] = path
            return path

    def fake_evaluate(
        gold,
        prediction,
    ):
        calls["gold"] = gold
        calls["prediction"] = prediction
        return FakeResult()

    monkeypatch.setattr(
        cli,
        "evaluate_bibliography",
        fake_evaluate,
    )

    cli.main()

    assert calls == {
        "gold": Path("gold.json"),
        "prediction": Path("bibliography.json"),
        "out": Path("evaluation.json"),
    }
