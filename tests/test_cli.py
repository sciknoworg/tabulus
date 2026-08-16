from pathlib import Path

import tabulus.cli as cli


def test_profile_parser_accepts_pipeline():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "profile",
            "--pdf",
            "paper.pdf",
            "--out",
            "output",
            "--backend",
            "pipeline",
        ]
    )

    assert args.command == "profile"
    assert args.pdf == Path("paper.pdf")
    assert args.out == Path("output")
    assert args.backend == "pipeline"
    assert args.method == "auto"
    assert args.effort == "high"


def test_profile_main_calls_mineru_runner(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "profile",
            "--pdf",
            "paper.pdf",
            "--out",
            "output",
            "--backend",
            "pipeline",
        ],
    )

    monkeypatch.setattr(
        cli,
        "select_backend",
        lambda requested: "pipeline",
    )

    def fake_run_mineru(
        pdf_path,
        output_dir,
        *,
        requested_backend,
        effort,
        method,
    ):
        calls["pdf_path"] = pdf_path
        calls["output_dir"] = output_dir
        calls["backend"] = requested_backend
        calls["effort"] = effort
        calls["method"] = method

    monkeypatch.setattr(
        cli,
        "run_mineru",
        fake_run_mineru,
    )

    cli.main()

    assert calls["pdf_path"] == Path("paper.pdf")
    assert calls["output_dir"] == Path("output")
    assert calls["backend"] == "pipeline"
    assert calls["effort"] == "high"
    assert calls["method"] == "auto"
