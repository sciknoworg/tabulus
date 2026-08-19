from pathlib import Path

from tabulus import cli


def _make_crop_root(parent: Path, name: str) -> Path:
    root = parent / name
    root.mkdir()
    (root / "tables_index.json").write_text(
        "{}",
        encoding="utf-8",
    )
    return root


def test_reconstruct_parser_accepts_crops_folder() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "reconstruct-tables",
            "--crops-folder",
            "table-crops",
            "--device",
            "gpu:0",
        ]
    )

    assert args.crops is None
    assert args.crops_folder == Path("table-crops")
    assert args.crops_list is None


def test_reconstruct_parser_accepts_crops_list() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "reconstruct-tables",
            "--crops-list",
            "crops.txt",
        ]
    )

    assert args.crops is None
    assert args.crops_folder is None
    assert args.crops_list == Path("crops.txt")


def test_reconstruct_folder_batch_reuses_adapter_and_outputs_per_paper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crops_parent = tmp_path / "table-crops"
    crops_parent.mkdir()
    alpha = _make_crop_root(crops_parent, "Alpha")
    zeta = _make_crop_root(crops_parent, "zeta")

    calls = {
        "create": [],
        "runs": [],
    }

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "reconstruct-tables",
            "--crops-folder",
            str(crops_parent),
            "--adapter",
            "paddleocr-vl",
            "--device",
            "gpu:0",
        ],
    )

    class FakeCapabilities:
        def supports_device(self, device):
            return device == "gpu:0"

    class FakeAdapter:
        capabilities = FakeCapabilities()

    fake_adapter = FakeAdapter()

    def fake_create_adapter(name, **kwargs):
        calls["create"].append((name, kwargs))
        return fake_adapter

    class FakeBatchResult:
        tables_requested = 2
        tables_ok = 2
        tables_empty = 0
        tables_error = 0
        prediction_csvs = 2

        def __init__(self, summary_path):
            self.summary_path = summary_path

    def fake_run_batch(*, crop_root, output_dir, adapter):
        calls["runs"].append(
            (crop_root, output_dir, adapter)
        )
        return FakeBatchResult(
            output_dir / "batch_summary.json"
        )

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

    assert calls["create"] == [
        ("paddleocr-vl", {"device": "gpu:0"})
    ]
    assert [call[0] for call in calls["runs"]] == [
        alpha,
        zeta,
    ]
    assert [call[1] for call in calls["runs"]] == [
        alpha / "reconstructions/paddleocr-vl",
        zeta / "reconstructions/paddleocr-vl",
    ]
    assert all(
        call[2] is fake_adapter
        for call in calls["runs"]
    )
