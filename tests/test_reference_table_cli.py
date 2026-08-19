from __future__ import annotations

from pathlib import Path

import tabulus.cli as cli


def test_default_reference_table_classification_output() -> None:
    reconstruction = Path("work/reconstructions/paddleocr-vl")

    output = cli.default_reference_table_classification_output(
        reconstruction
    )

    assert output == (
        reconstruction / "reference_table_classification.json"
    )


def test_classify_reference_tables_parser_accepts_reconstruction() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "classify-reference-tables",
            "--reconstruction",
            "work/reconstructions/paddleocr-vl",
        ]
    )

    assert args.command == "classify-reference-tables"
    assert args.reconstruction == Path(
        "work/reconstructions/paddleocr-vl"
    )
    assert args.out is None


def test_classify_reference_tables_main_dispatches(monkeypatch) -> None:
    calls = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "classify-reference-tables",
            "--reconstruction",
            "work/reconstructions/paddleocr-vl",
        ],
    )

    class FakeResult:
        tables_considered = 23
        reference_tables_found = 4
        output_path = Path(
            "work/reconstructions/paddleocr-vl/"
            "reference_table_classification.json"
        )

    def fake_classify(reconstruction_dir, *, output_path=None):
        calls["reconstruction_dir"] = reconstruction_dir
        calls["output_path"] = output_path
        return FakeResult()

    monkeypatch.setattr(
        cli,
        "classify_reconstruction_tables",
        fake_classify,
    )

    cli.main()

    assert calls["reconstruction_dir"] == Path(
        "work/reconstructions/paddleocr-vl"
    )
    assert calls["output_path"] == Path(
        "work/reconstructions/paddleocr-vl/"
        "reference_table_classification.json"
    )
