from __future__ import annotations

from pathlib import Path

import tabulus.cli as cli


def test_evaluate_table_reconstruction_parser_accepts_csv_pair() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "evaluate-table-reconstruction",
            "--gold",
            "gold.csv",
            "--prediction",
            "prediction.csv",
            "--out",
            "evaluation.json",
        ]
    )

    assert args.command == "evaluate-table-reconstruction"
    assert args.gold == Path("gold.csv")
    assert args.prediction == Path("prediction.csv")
    assert args.metric == "rms"
    assert args.text_threshold == 0.5
    assert args.number_threshold == 0.1
    assert args.out == Path("evaluation.json")


def test_evaluate_table_reconstruction_main_dispatches(monkeypatch) -> None:
    calls = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "evaluate-table-reconstruction",
            "--gold",
            "gold.csv",
            "--prediction",
            "prediction.csv",
            "--text-threshold",
            "0.4",
            "--number-threshold",
            "0.05",
            "--out",
            "evaluation.json",
        ],
    )

    class FakeResult:
        gold_csv = Path("gold.csv")
        prediction_csv = Path("prediction.csv")
        metric_name = "Relative Mapping Similarity"
        metric_short_name = "RMS"
        score_scale = "[0,100]"
        precision = 90.0
        recall = 80.0
        f1 = 84.70588235

        def write_json(self, path):
            calls["out"] = path
            return path

    def fake_evaluate(
        gold_csv,
        prediction_csv,
        *,
        metric,
        text_threshold,
        number_threshold,
    ):
        calls["gold_csv"] = gold_csv
        calls["prediction_csv"] = prediction_csv
        calls["metric"] = metric
        calls["text_threshold"] = text_threshold
        calls["number_threshold"] = number_threshold
        return FakeResult()

    monkeypatch.setattr(cli, "evaluate_table_reconstruction", fake_evaluate)

    cli.main()

    assert calls == {
        "gold_csv": Path("gold.csv"),
        "prediction_csv": Path("prediction.csv"),
        "metric": "rms",
        "text_threshold": 0.4,
        "number_threshold": 0.05,
        "out": Path("evaluation.json"),
    }
