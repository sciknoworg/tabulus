from __future__ import annotations

from pathlib import Path


def _validate_pdf_file(path: Path, *, source: str) -> Path:
    path = Path(path).expanduser()

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"{source} is not a PDF file: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"{source} not found: {path}")

    return path


def _deduplicate_pdf_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []

    for path in paths:
        key = path.resolve()

        if key in seen:
            raise ValueError(f"Duplicate PDF input: {path}")

        seen.add(key)
        result.append(path)

    return result


def _pdfs_from_folder(folder: Path) -> list[Path]:
    folder = Path(folder).expanduser()

    if not folder.is_dir():
        raise NotADirectoryError(f"PDF folder not found: {folder}")

    pdfs = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )

    if not pdfs:
        raise ValueError(f"No PDF files found in folder: {folder}")

    return pdfs


def _pdfs_from_list(pdf_list: Path) -> list[Path]:
    pdf_list = Path(pdf_list).expanduser()

    if not pdf_list.is_file():
        raise FileNotFoundError(f"PDF list file not found: {pdf_list}")

    pdfs: list[Path] = []

    for line_number, raw_line in enumerate(
        pdf_list.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = raw_line.strip()

        if not value or value.startswith("#"):
            continue

        path = Path(value).expanduser()

        if not path.is_absolute():
            path = pdf_list.parent / path

        try:
            pdfs.append(
                _validate_pdf_file(
                    path,
                    source=f"PDF list line {line_number}",
                )
            )
        except (ValueError, FileNotFoundError) as error:
            raise type(error)(
                f"{error} (list file: {pdf_list})"
            ) from error

    if not pdfs:
        raise ValueError(f"No PDF paths found in list file: {pdf_list}")

    return _deduplicate_pdf_paths(pdfs)


def resolve_pdf_inputs(
    *,
    pdf: Path | None = None,
    folder: Path | None = None,
    pdf_list: Path | None = None,
) -> list[Path]:
    """
    Resolve one CLI PDF-input mode into an ordered list of PDF paths.

    Exactly one input mode must be supplied:
    - ``pdf``: one explicitly named PDF;
    - ``folder``: all PDFs directly inside a folder, sorted by filename;
    - ``pdf_list``: one PDF path per non-empty, non-comment line.

    Relative paths in a list file are resolved relative to that list file.
    Folder discovery is intentionally non-recursive.
    """

    supplied = sum(
        value is not None
        for value in (pdf, folder, pdf_list)
    )

    if supplied != 1:
        raise ValueError(
            "Exactly one PDF input source is required: "
            "pdf, folder, or pdf_list."
        )

    if pdf is not None:
        # Keep the existing single-PDF CLI behavior: the downstream profiler
        # remains responsible for reporting a missing explicitly named file.
        path = Path(pdf).expanduser()

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Input is not a PDF file: {path}")

        return [path]

    if folder is not None:
        return _deduplicate_pdf_paths(_pdfs_from_folder(folder))

    assert pdf_list is not None
    return _pdfs_from_list(pdf_list)
