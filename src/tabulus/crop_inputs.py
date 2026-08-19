from __future__ import annotations

from pathlib import Path


TABLES_INDEX_NAME = "tables_index.json"


def _validate_crop_root(path: Path, *, source: str) -> Path:
    path = Path(path).expanduser()

    if not path.is_dir():
        raise NotADirectoryError(f"{source} not found: {path}")

    index_path = path / TABLES_INDEX_NAME

    if not index_path.is_file():
        raise FileNotFoundError(
            f"{source} has no {TABLES_INDEX_NAME}: {path}"
        )

    return path


def _deduplicate_crop_roots(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []

    for path in paths:
        key = path.resolve()

        if key in seen:
            raise ValueError(f"Duplicate table-crop input: {path}")

        seen.add(key)
        result.append(path)

    return result


def _crop_roots_from_folder(folder: Path) -> list[Path]:
    folder = Path(folder).expanduser()

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Table-crops folder not found: {folder}"
        )

    crop_roots = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_dir()
            and (path / TABLES_INDEX_NAME).is_file()
        ),
        key=lambda path: path.name.casefold(),
    )

    if not crop_roots:
        raise ValueError(
            "No canonical table-crop directories containing "
            f"{TABLES_INDEX_NAME} found in: {folder}"
        )

    return crop_roots


def _crop_roots_from_list(crops_list: Path) -> list[Path]:
    crops_list = Path(crops_list).expanduser()

    if not crops_list.is_file():
        raise FileNotFoundError(
            f"Table-crops list file not found: {crops_list}"
        )

    crop_roots: list[Path] = []

    for line_number, raw_line in enumerate(
        crops_list.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = raw_line.strip()

        if not value or value.startswith("#"):
            continue

        path = Path(value).expanduser()

        if not path.is_absolute():
            path = crops_list.parent / path

        try:
            crop_roots.append(
                _validate_crop_root(
                    path,
                    source=(
                        "Table-crops list line "
                        f"{line_number}"
                    ),
                )
            )
        except (NotADirectoryError, FileNotFoundError) as error:
            raise type(error)(
                f"{error} (list file: {crops_list})"
            ) from error

    if not crop_roots:
        raise ValueError(
            f"No table-crop paths found in list file: {crops_list}"
        )

    return _deduplicate_crop_roots(crop_roots)


def resolve_crop_inputs(
    *,
    crops: Path | None = None,
    crops_folder: Path | None = None,
    crops_list: Path | None = None,
) -> list[Path]:
    """
    Resolve one reconstruction input mode into ordered crop roots.

    Exactly one input mode must be supplied:
    - ``crops``: one canonical table-crop directory;
    - ``crops_folder``: immediate child directories containing
      ``tables_index.json``, sorted by directory name;
    - ``crops_list``: one canonical table-crop directory per non-empty,
      non-comment line.

    Relative paths in a list file are resolved relative to that list file.
    Folder discovery is intentionally non-recursive.
    """

    supplied = sum(
        value is not None
        for value in (crops, crops_folder, crops_list)
    )

    if supplied != 1:
        raise ValueError(
            "Exactly one table-crop input source is required: "
            "crops, crops_folder, or crops_list."
        )

    if crops is not None:
        # Preserve the established single-crop CLI behavior. The batch layer
        # remains responsible for validating an explicitly supplied crop root.
        return [Path(crops).expanduser()]

    if crops_folder is not None:
        return _deduplicate_crop_roots(
            _crop_roots_from_folder(crops_folder)
        )

    assert crops_list is not None
    return _crop_roots_from_list(crops_list)
