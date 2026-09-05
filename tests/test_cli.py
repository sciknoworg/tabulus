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

def test_default_table_crops_output_root(tmp_path):
    pdf = tmp_path / "Puurunen - February 2005.pdf"

    output = cli.default_table_crops_output_root(pdf)

    assert output == (
        tmp_path
        / "tabulus-output"
        / "table-crops"
        / "Puurunen - February 2005"
    )


def test_default_table_reconstruction_output_root(tmp_path):
    crops = tmp_path / "table-crops" / "paper"

    output = cli.default_table_reconstruction_output_root(
        crops,
        adapter_name="paddleocr-vl",
    )

    assert output == (
        crops
        / "reconstructions"
        / "paddleocr-vl"
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
    assert args.export_table_crops is True
    assert args.table_crops_out is None

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


def test_profile_parser_can_disable_table_crop_export():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "profile",
            "--pdf",
            "paper.pdf",
            "--backend",
            "pipeline",
            "--no-export-table-crops",
        ]
    )

    assert args.export_table_crops is False


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
            "--table-crops-out",
            "table-crops",
        ],
    )

    monkeypatch.setattr(
        cli,
        "select_backend",
        lambda requested: "pipeline",
    )

    run_dir = Path("output/paper/auto")

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
        return run_dir

    class FakeResult:
        tables_found = 2
        crops_saved = 2
        index_path = Path("table-crops/tables_index.json")

    def fake_export_mineru_table_crops(
        *,
        mineru_output_dir,
        output_dir,
    ):
        calls["mineru_output_dir"] = mineru_output_dir
        calls["table_crops_output_dir"] = output_dir
        return FakeResult()

    monkeypatch.setattr(
        cli,
        "run_mineru",
        fake_run_mineru,
    )
    monkeypatch.setattr(
        cli,
        "export_mineru_table_crops",
        fake_export_mineru_table_crops,
    )

    cli.main()

    assert calls["pdf_path"] == Path("paper.pdf")
    assert calls["output_dir"] == Path("output")
    assert calls["backend"] == "pipeline"
    assert calls["effort"] == "high"
    assert calls["method"] == "auto"
    assert calls["mineru_output_dir"] == run_dir
    assert calls["table_crops_output_dir"] == Path("table-crops")


def test_profile_main_can_skip_table_crop_export(monkeypatch):
    calls = {"export_called": False}

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
            "--no-export-table-crops",
        ],
    )

    monkeypatch.setattr(
        cli,
        "select_backend",
        lambda requested: "pipeline",
    )

    monkeypatch.setattr(
        cli,
        "run_mineru",
        lambda *args, **kwargs: Path("output/paper/auto"),
    )

    def fake_export_mineru_table_crops(**kwargs):
        calls["export_called"] = True
        raise AssertionError("Table crop export should have been skipped.")

    monkeypatch.setattr(
        cli,
        "export_mineru_table_crops",
        fake_export_mineru_table_crops,
    )

    cli.main()

    assert calls["export_called"] is False


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


def test_reconstruct_tables_parser_uses_default_adapter_and_output():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "reconstruct-tables",
            "--crops",
            "work/table-crops/paper",
            "--device",
            "gpu:0",
        ]
    )

    assert args.command == "reconstruct-tables"
    assert args.crops == Path("work/table-crops/paper")
    assert args.adapter == "paddleocr-vl"
    assert args.device == "gpu:0"
    assert args.out is None


def test_reconstruct_tables_main_dispatches_to_batch_layer(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "reconstruct-tables",
            "--crops",
            "work/table-crops/paper",
            "--adapter",
            "paddleocr-vl",
            "--device",
            "gpu:0",
        ],
    )

    class FakeCapabilities:
        def supports_device(self, device):
            calls["supports_device"] = device
            return True

    class FakeAdapter:
        capabilities = FakeCapabilities()

    fake_adapter = FakeAdapter()

    def fake_create_adapter(name, **kwargs):
        calls["adapter_name"] = name
        calls["adapter_kwargs"] = kwargs
        return fake_adapter

    class FakeBatchResult:
        tables_requested = 23
        tables_ok = 23
        tables_empty = 0
        tables_error = 0
        prediction_csvs = 23
        summary_path = Path(
            "work/table-crops/paper/reconstructions/"
            "paddleocr-vl/batch_summary.json"
        )

    def fake_run_batch(*, crop_root, output_dir, adapter):
        calls["crop_root"] = crop_root
        calls["output_dir"] = output_dir
        calls["adapter"] = adapter
        return FakeBatchResult()

    monkeypatch.setattr(
        cli,
        "create_table_ocr_adapter",
        fake_create_adapter,
    )
    monkeypatch.setattr(
        cli,
        "run_table_ocr_batch",
        fake_run_batch,
    )

    cli.main()

    assert calls["adapter_name"] == "paddleocr-vl"
    assert calls["adapter_kwargs"] == {"device": "gpu:0"}
    assert calls["supports_device"] == "gpu:0"
    assert calls["crop_root"] == Path("work/table-crops/paper")
    assert calls["output_dir"] == Path(
        "work/table-crops/paper/reconstructions/paddleocr-vl"
    )
    assert calls["adapter"] is fake_adapter


def test_match_references_parser_accepts_stage_artifacts():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "match-references",
            "--selected",
            "reconstruction/selected_reference_tables.json",
            "--bibliography",
            "references/bibliography.json",
        ]
    )

    assert args.command == "match-references"
    assert args.selected == Path(
        "reconstruction/selected_reference_tables.json"
    )
    assert args.bibliography == Path(
        "references/bibliography.json"
    )
    assert args.out is None


def test_match_references_main_calls_pipeline(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "match-references",
            "--selected",
            "reconstruction/selected_reference_tables.json",
            "--bibliography",
            "references/bibliography.json",
            "--out",
            "reference_matches.json",
        ],
    )

    class FakeResult:
        reference_tables_selected = 3
        reference_tables_checked = 2
        reference_tables_skipped = 1
        output_path = Path("reference_matches.json")

    def fake_match_selected_reference_tables(
        selected,
        bibliography,
        *,
        output_path=None,
    ):
        calls["selected"] = selected
        calls["bibliography"] = bibliography
        calls["output_path"] = output_path
        return FakeResult()

    monkeypatch.setattr(
        cli,
        "match_selected_reference_tables",
        fake_match_selected_reference_tables,
    )

    cli.main()

    assert calls["selected"] == Path(
        "reconstruction/selected_reference_tables.json"
    )
    assert calls["bibliography"] == Path(
        "references/bibliography.json"
    )
    assert calls["output_path"] == Path(
        "reference_matches.json"
    )
