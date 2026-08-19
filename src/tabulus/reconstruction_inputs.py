from __future__ import annotations

from pathlib import Path

from tabulus.crop_inputs import resolve_crop_inputs


BATCH_SUMMARY_NAME = "batch_summary.json"


def _validate_reconstruction_dir(
    path: Path,
    *,
    source: str,
) -> Path:
    path = Path(path).expanduser()

    if not path.is_dir():
        raise NotADirectoryError(f"{source} not found: {path}")

    summary_path = path / BATCH_SUMMARY_NAME

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{source} has no {BATCH_SUMMARY_NAME}: {path}"
        )

    return path


def _deduplicate_reconstruction_dirs(
    paths: list[Path],
) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []

    for path in paths:
        key = path.resolve()

        if key in seen:
            raise ValueError(f"Duplicate reconstruction input: {path}")

        seen.add(key)
        result.append(path)

    return result


def _reconstructions_from_crops_folder(
    crops_folder: Path,
    *,
    adapter_name: str,
) -> list[Path]:
    crop_roots = resolve_crop_inputs(crops_folder=crops_folder)
    reconstructions: list[Path] = []

    for crop_root in crop_roots:
        reconstruction_dir = (
            crop_root / "reconstructions" / adapter_name
        )
        reconstructions.append(
            _validate_reconstruction_dir(
                reconstruction_dir,
                source=(
                    f"Reconstruction for crop root {crop_root.name!r} "
                    f"and adapter {adapter_name!r}"
                ),
            )
        )

    return _deduplicate_reconstruction_dirs(reconstructions)


def _reconstructions_from_list(
    reconstruction_list: Path,
) -> list[Path]:
    reconstruction_list = Path(reconstruction_list).expanduser()

    if not reconstruction_list.is_file():
        raise FileNotFoundError(
            f"Reconstruction list file not found: {reconstruction_list}"
        )

    reconstructions: list[Path] = []

    for line_number, raw_line in enumerate(
        reconstruction_list.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = raw_line.strip()

        if not value or value.startswith("#"):
            continue

        path = Path(value).expanduser()

        if not path.is_absolute():
            path = reconstruction_list.parent / path

        try:
            reconstructions.append(
                _validate_reconstruction_dir(
                    path,
                    source=f"Reconstruction list line {line_number}",
                )
            )
        except (NotADirectoryError, FileNotFoundError) as error:
            raise type(error)(
                f"{error} (list file: {reconstruction_list})"
            ) from error

    if not reconstructions:
        raise ValueError(
            f"No reconstruction paths found in list file: "
            f"{reconstruction_list}"
        )

    return _deduplicate_reconstruction_dirs(reconstructions)


def resolve_reconstruction_inputs(
    *,
    reconstruction: Path | None = None,
    crops_folder: Path | None = None,
    reconstruction_list: Path | None = None,
    adapter_name: str = "paddleocr-vl",
) -> list[Path]:
    """Resolve one reference-classification input mode.

    Exactly one input mode must be supplied:
    - ``reconstruction``: one reconstruction directory;
    - ``crops_folder``: canonical crop roots beneath one parent, resolved
      to ``reconstructions/<adapter_name>/``;
    - ``reconstruction_list``: one reconstruction directory per
      non-empty, non-comment line.

    Folder discovery is non-recursive. Relative paths in a list file are
    resolved relative to that list file.
    """

    supplied = sum(
        value is not None
        for value in (
            reconstruction,
            crops_folder,
            reconstruction_list,
        )
    )

    if supplied != 1:
        raise ValueError(
            "Exactly one reconstruction input source is required: "
            "reconstruction, crops_folder, or reconstruction_list."
        )

    if reconstruction is not None:
        # Preserve the established single-input contract. The classifier
        # remains responsible for validating an explicitly supplied path.
        return [Path(reconstruction).expanduser()]

    if crops_folder is not None:
        return _reconstructions_from_crops_folder(
            crops_folder,
            adapter_name=adapter_name,
        )

    assert reconstruction_list is not None
    return _reconstructions_from_list(reconstruction_list)
