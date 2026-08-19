from pathlib import Path

import pytest

from tabulus.crop_inputs import resolve_crop_inputs


def _make_crop_root(parent: Path, name: str) -> Path:
    root = parent / name
    root.mkdir()
    (root / "tables_index.json").write_text(
        "{}",
        encoding="utf-8",
    )
    return root


def test_single_crop_input_is_preserved() -> None:
    assert resolve_crop_inputs(crops=Path("paper")) == [
        Path("paper")
    ]


def test_folder_discovers_canonical_crop_roots_and_sorts(
    tmp_path: Path,
) -> None:
    zeta = _make_crop_root(tmp_path, "zeta")
    alpha = _make_crop_root(tmp_path, "Alpha")
    ignored = tmp_path / "not-a-crop-root"
    ignored.mkdir()

    assert resolve_crop_inputs(crops_folder=tmp_path) == [
        alpha,
        zeta,
    ]


def test_folder_discovery_is_non_recursive(tmp_path: Path) -> None:
    nested_parent = tmp_path / "nested"
    nested_parent.mkdir()
    _make_crop_root(nested_parent, "paper")

    with pytest.raises(ValueError, match="No canonical table-crop"):
        resolve_crop_inputs(crops_folder=tmp_path)


def test_crop_list_supports_comments_and_relative_paths(
    tmp_path: Path,
) -> None:
    first = _make_crop_root(tmp_path, "first")
    second = _make_crop_root(tmp_path, "second")
    crops_list = tmp_path / "crops.txt"
    crops_list.write_text(
        "# selected papers\n"
        "\n"
        "first\n"
        "second\n",
        encoding="utf-8",
    )

    assert resolve_crop_inputs(crops_list=crops_list) == [
        first,
        second,
    ]


def test_crop_list_rejects_duplicate_inputs(tmp_path: Path) -> None:
    _make_crop_root(tmp_path, "paper")
    crops_list = tmp_path / "crops.txt"
    crops_list.write_text(
        "paper\n./paper\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate table-crop input"):
        resolve_crop_inputs(crops_list=crops_list)


def test_exactly_one_crop_input_mode_is_required(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Exactly one table-crop input source",
    ):
        resolve_crop_inputs()

    with pytest.raises(
        ValueError,
        match="Exactly one table-crop input source",
    ):
        resolve_crop_inputs(
            crops=Path("paper"),
            crops_folder=tmp_path,
        )
