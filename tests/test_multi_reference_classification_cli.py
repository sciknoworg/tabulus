from pathlib import Path

import pytest

from tabulus import cli


def _make_crop_reconstruction(
    parent: Path,
    paper: str,
) -> Path:
    crop_root = parent / paper
    crop_root.mkdir()
    (crop_root / "tables_index.json").write_text(
        "{}",
        encoding="utf-8",
    )
    reconstruction = (
        crop_root / "reconstructions" / "paddleocr-vl"
    )
    reconstruction.mkdir(parents=True)
    (reconstruction / "batch_summary.json").write_text(
        "{}",
        encoding="utf-8",
    )
    return reconstruction


def test_classification_parser_accepts_crops_folder() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "classify-reference-tables",
            "--crops-folder",
            "table-crops",
            "--adapter",
            "paddleocr-vl",
        ]
    )

    assert args.reconstruction is None
    assert args.crops_folder == Path("table-crops")
    assert args.reconstruction_list is None
    assert args.adapter == "paddleocr-vl"


def test_classification_parser_accepts_reconstruction_list() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "classify-reference-tables",
            "--reconstruction-list",
            "reconstructions.txt",
        ]
    )

    assert args.reconstruction is None
    assert args.crops_folder is None
    assert args.reconstruction_list == Path("reconstructions.txt")


def test_classification_folder_batch_writes_per_reconstruction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop_parent = tmp_path / "table-crops"
    crop_parent.mkdir()
    alpha = _make_crop_reconstruction(crop_parent, "Alpha")
    zeta = _make_crop_reconstruction(crop_parent, "zeta")

    calls = []

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "classify-reference-tables",
            "--crops-folder",
            str(crop_parent),
            "--adapter",
            "paddleocr-vl",
        ],
    )

    class FakeResult:
        def __init__(
            self,
            reconstruction_dir: Path,
            *,
            tables_considered: int,
            reference_tables_found: int,
        ):
            self.tables_considered = tables_considered
            self.reference_tables_found = reference_tables_found
            self.output_path = (
                reconstruction_dir
                / "reference_table_classification.json"
            )

    def fake_classify(reconstruction_dir, *, output_path):
        calls.append((reconstruction_dir, output_path))
        if reconstruction_dir == alpha:
            return FakeResult(
                reconstruction_dir,
                tables_considered=2,
                reference_tables_found=1,
            )
        return FakeResult(
            reconstruction_dir,
            tables_considered=3,
            reference_tables_found=2,
        )

    monkeypatch.setattr(
        cli,
        "classify_reconstruction_tables",
        fake_classify,
    )

    cli.main()

    assert [call[0] for call in calls] == [alpha, zeta]
    assert [call[1] for call in calls] == [
        alpha / "reference_table_classification.json",
        zeta / "reference_table_classification.json",
    ]


def test_classification_rejects_custom_out_for_multiple_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop_parent = tmp_path / "table-crops"
    crop_parent.mkdir()
    _make_crop_reconstruction(crop_parent, "one")
    _make_crop_reconstruction(crop_parent, "two")

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "classify-reference-tables",
            "--crops-folder",
            str(crop_parent),
            "--out",
            str(tmp_path / "classification.json"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="--out can only be used",
    ):
        cli.main()
