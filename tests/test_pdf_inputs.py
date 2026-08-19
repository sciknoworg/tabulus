from pathlib import Path

import pytest

from tabulus.pdf_inputs import resolve_pdf_inputs


def test_single_pdf_input_is_preserved() -> None:
    assert resolve_pdf_inputs(pdf=Path("paper.pdf")) == [
        Path("paper.pdf")
    ]


def test_folder_discovers_pdfs_non_recursively_and_sorts(
    tmp_path: Path,
) -> None:
    (tmp_path / "zeta.PDF").write_bytes(b"%PDF")
    (tmp_path / "Alpha.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.pdf").write_bytes(b"%PDF")

    assert resolve_pdf_inputs(folder=tmp_path) == [
        tmp_path / "Alpha.pdf",
        tmp_path / "zeta.PDF",
    ]


def test_folder_without_pdfs_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No PDF files found"):
        resolve_pdf_inputs(folder=tmp_path)


def test_pdf_list_supports_comments_blank_lines_and_relative_paths(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF")
    second.write_bytes(b"%PDF")

    pdf_list = tmp_path / "papers.txt"
    pdf_list.write_text(
        "# papers for this run\n"
        "\n"
        "first.pdf\n"
        "second.pdf\n",
        encoding="utf-8",
    )

    assert resolve_pdf_inputs(pdf_list=pdf_list) == [
        first,
        second,
    ]


def test_pdf_list_rejects_duplicate_inputs(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    pdf_list = tmp_path / "papers.txt"
    pdf_list.write_text(
        "paper.pdf\n./paper.pdf\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate PDF input"):
        resolve_pdf_inputs(pdf_list=pdf_list)


def test_exactly_one_input_mode_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Exactly one PDF input source"):
        resolve_pdf_inputs()

    with pytest.raises(ValueError, match="Exactly one PDF input source"):
        resolve_pdf_inputs(
            pdf=Path("paper.pdf"),
            folder=tmp_path,
        )
