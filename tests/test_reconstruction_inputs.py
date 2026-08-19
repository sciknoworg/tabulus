from pathlib import Path

import pytest

from tabulus.reconstruction_inputs import resolve_reconstruction_inputs


def _make_crop_reconstruction(
    parent: Path,
    paper: str,
    *,
    adapter: str = "paddleocr-vl",
) -> Path:
    crop_root = parent / paper
    crop_root.mkdir()
    (crop_root / "tables_index.json").write_text(
        "{}",
        encoding="utf-8",
    )
    reconstruction = crop_root / "reconstructions" / adapter
    reconstruction.mkdir(parents=True)
    (reconstruction / "batch_summary.json").write_text(
        "{}",
        encoding="utf-8",
    )
    return reconstruction


def test_single_reconstruction_is_preserved() -> None:
    assert resolve_reconstruction_inputs(
        reconstruction=Path("reconstruction")
    ) == [Path("reconstruction")]


def test_crops_folder_resolves_selected_adapter_and_sorts(
    tmp_path: Path,
) -> None:
    zeta = _make_crop_reconstruction(tmp_path, "zeta")
    alpha = _make_crop_reconstruction(tmp_path, "Alpha")

    assert resolve_reconstruction_inputs(
        crops_folder=tmp_path,
        adapter_name="paddleocr-vl",
    ) == [alpha, zeta]


def test_crops_folder_rejects_missing_selected_adapter(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "paper"
    crop_root.mkdir()
    (crop_root / "tables_index.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        NotADirectoryError,
        match="not found",
    ):
        resolve_reconstruction_inputs(
            crops_folder=tmp_path,
            adapter_name="paddleocr-vl",
        )

def test_crops_folder_rejects_missing_batch_summary(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "paper"
    crop_root.mkdir()
    (crop_root / "tables_index.json").write_text(
        "{}",
        encoding="utf-8",
    )

    reconstruction = (
        crop_root
        / "reconstructions"
        / "paddleocr-vl"
    )
    reconstruction.mkdir(parents=True)

    with pytest.raises(
        FileNotFoundError,
        match="has no batch_summary.json",
    ):
        resolve_reconstruction_inputs(
            crops_folder=tmp_path,
            adapter_name="paddleocr-vl",
        )

def test_reconstruction_list_supports_comments_and_relative_paths(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    for path in (first, second):
        path.mkdir()
        (path / "batch_summary.json").write_text(
            "{}",
            encoding="utf-8",
        )

    reconstruction_list = tmp_path / "reconstructions.txt"
    reconstruction_list.write_text(
        "# selected reconstruction directories\n"
        "\n"
        "first\n"
        "second\n",
        encoding="utf-8",
    )

    assert resolve_reconstruction_inputs(
        reconstruction_list=reconstruction_list
    ) == [first, second]


def test_reconstruction_list_rejects_duplicates(
    tmp_path: Path,
) -> None:
    reconstruction = tmp_path / "paper"
    reconstruction.mkdir()
    (reconstruction / "batch_summary.json").write_text(
        "{}",
        encoding="utf-8",
    )
    reconstruction_list = tmp_path / "reconstructions.txt"
    reconstruction_list.write_text(
        "paper\n./paper\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate reconstruction input",
    ):
        resolve_reconstruction_inputs(
            reconstruction_list=reconstruction_list
        )


def test_exactly_one_input_mode_is_required(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Exactly one reconstruction input source",
    ):
        resolve_reconstruction_inputs()

    with pytest.raises(
        ValueError,
        match="Exactly one reconstruction input source",
    ):
        resolve_reconstruction_inputs(
            reconstruction=Path("one"),
            crops_folder=tmp_path,
        )
