from __future__ import annotations

from tabulus.table_ocr.parsing import (
    otsl_table_to_html,
    parse_table_text,
)


def test_otsl_simple_table_enters_shared_html_parser() -> None:
    otsl = (
        "<fcel>Z<fcel>Material<fcel>References<nl>"
        "<fcel>13<fcel>Al2O3<fcel>103<nl>"
    )

    html = otsl_table_to_html(otsl)
    tables = parse_table_text(html)

    assert len(tables) == 1
    assert tables[0].source == "html"
    assert tables[0].rows == [
        ["Z", "Material", "References"],
        ["13", "Al2O3", "103"],
    ]


def test_otsl_empty_cells_are_preserved() -> None:
    otsl = (
        "<fcel>A<fcel>B<fcel>C<nl>"
        "<ecel><fcel>2<ecel><nl>"
    )

    rows = parse_table_text(
        otsl_table_to_html(otsl)
    )[0].rows

    assert rows == [
        ["A", "B", "C"],
        ["", "2", ""],
    ]


def test_otsl_rowspan_is_preserved() -> None:
    otsl = (
        "<fcel>Material<fcel>Reactant<nl>"
        "<fcel>AlN<fcel>NH3<nl>"
        "<ucel><fcel>N2<nl>"
    )

    html = otsl_table_to_html(otsl)

    assert 'rowspan="2"' in html
    assert parse_table_text(html)[0].rows == [
        ["Material", "Reactant"],
        ["AlN", "NH3"],
        ["", "N2"],
    ]


def test_otsl_colspan_is_preserved() -> None:
    otsl = (
        "<fcel>Group<lcel><fcel>Reference<nl>"
        "<fcel>A<fcel>B<fcel>1<nl>"
    )

    html = otsl_table_to_html(otsl)

    assert 'colspan="2"' in html
    assert parse_table_text(html)[0].rows == [
        ["Group", "", "Reference"],
        ["A", "B", "1"],
    ]


def test_otsl_cell_text_is_html_escaped() -> None:
    otsl = "<fcel>A&B<fcel><x><nl>"

    html = otsl_table_to_html(otsl)

    assert "A&amp;B" in html
    assert "&lt;x&gt;" in html


def test_otsl_ragged_rows_use_widest_generated_row() -> None:
    otsl = (
        "<fcel>A<fcel>B<fcel>C<nl>"
        "<fcel>1<fcel>2<fcel>3<fcel>4<nl>"
    )

    rows = parse_table_text(
        otsl_table_to_html(otsl)
    )[0].rows

    assert rows == [
        ["A", "B", "C", ""],
        ["1", "2", "3", "4"],
    ]
