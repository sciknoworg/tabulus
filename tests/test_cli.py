from pathlib import Path

import tabulus.cli as cli

def test_default_profile_output_root(tmp_path):
    pdf = tmp_path / "Puurunen - February 2005.pdf"

    output = cli.default_profile_output_root(
        pdf,
        profiler="mineru",
        backend="pipeline",
    )

    assert output == (
        tmp_path
        / "tabulus-output"
        / "mineru"
        / "pipeline"
    )

def test_profile_parser_allows_default_output():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "profile",
            "--pdf",
            "paper.pdf",
            "--backend",
            "pipeline",
        ]
    )

    assert args.out is None
    assert args.profiler == "mineru"

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


def test_export_table_crops_parser_accepts_mineru_root():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "export-table-crops",
            "--mineru-root",
            "work/mineru/puurunen_2005",
            "--out",
            "work/table_crops",
        ]
    )

    assert args.command == "export-table-crops"
    assert args.mineru_root == Path("work/mineru/puurunen_2005")
    assert args.out == Path("work/table_crops")


def test_export_table_crops_main_calls_exporter(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "export-table-crops",
            "--mineru-root",
            "work/mineru/puurunen_2005",
            "--out",
            "work/table_crops",
        ],
    )

    class FakeResult:
        tables_found = 2
        crops_saved = 2
        index_path = Path("work/table_crops/tables_index.json")

    def fake_export_mineru_table_crops(
        *,
        mineru_output_dir,
        output_dir,
    ):
        calls["mineru_output_dir"] = mineru_output_dir
        calls["output_dir"] = output_dir
        return FakeResult()

    monkeypatch.setattr(
        cli,
        "export_mineru_table_crops",
        fake_export_mineru_table_crops,
    )

    cli.main()

    assert calls["mineru_output_dir"] == Path("work/mineru/puurunen_2005")
    assert calls["output_dir"] == Path("work/table_crops")
