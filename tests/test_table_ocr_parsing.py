from tabulus.table_ocr.parsing import (
    extract_markdown_text,
    html_table_to_rows,
    markdown_table_to_rows,
    parse_native_markdown,
    parse_table_text,
)


def test_html_table_to_rows_preserves_cells_and_pads_rows() -> None:
    html = """
    <table>
      <tr><th>A</th><th>B</th><th>C</th></tr>
      <tr><td>$ H_{2}O $</td><td>value</td></tr>
    </table>
    """

    assert html_table_to_rows(html) == [
        ["A", "B", "C"],
        ["$ H_{2}O $", "value", ""],
    ]


def test_html_table_to_rows_expands_rowspan_without_shifting_cells() -> None:
    html = """
    <table>
      <tr>
        <th>Z</th>
        <th>Material</th>
        <th>Reactant A</th>
        <th>Reactant B</th>
        <th>Substrate</th>
        <th>Refs.</th>
      </tr>
      <tr>
        <td rowspan="3">5 Boron</td>
        <td>B2O3</td>
        <td>BBr3</td>
        <td>H2O</td>
        <td>SiO2 gel</td>
        <td>85</td>
      </tr>
      <tr>
        <td>BxPyOz</td>
        <td>B(OMe)3</td>
        <td>POCl3</td>
        <td>SiO2 gel</td>
        <td>88 and 89</td>
      </tr>
      <tr>
        <td></td><td></td><td></td><td></td><td></td>
      </tr>
    </table>
    """

    assert html_table_to_rows(html) == [
        ["Z", "Material", "Reactant A", "Reactant B", "Substrate", "Refs."],
        ["5 Boron", "B2O3", "BBr3", "H2O", "SiO2 gel", "85"],
        ["", "BxPyOz", "B(OMe)3", "POCl3", "SiO2 gel", "88 and 89"],
        ["", "", "", "", "", ""],
    ]


def test_html_table_to_rows_expands_colspan_with_empty_placeholders() -> None:
    html = """
    <table>
      <tr><th colspan="2">Material</th><th>Refs.</th></tr>
      <tr><td>A</td><td>B</td><td>10</td></tr>
    </table>
    """

    assert html_table_to_rows(html) == [
        ["Material", "", "Refs."],
        ["A", "B", "10"],
    ]


def test_html_table_to_rows_defaults_invalid_spans_to_one() -> None:
    html = """
    <table>
      <tr><td rowspan="bad">A</td><td colspan="0">B</td></tr>
    </table>
    """

    assert html_table_to_rows(html) == [["A", "B"]]


def test_parse_native_markdown_reads_paddle_markdown_texts() -> None:
    native = {
        "markdown_texts": """
        <table>
          <tr><td>$ Z^{a} $</td><td>Material</td></tr>
          <tr><td>5 Boron</td><td></td></tr>
        </table>
        """
    }

    tables = parse_native_markdown(native)

    assert len(tables) == 1
    assert tables[0].source == "html"
    assert tables[0].n_rows == 2
    assert tables[0].n_cols == 2
    assert tables[0].rows == [
        ["$ Z^{a} $", "Material"],
        ["5 Boron", ""],
    ]


def test_html_is_preferred_over_markdown_fallback() -> None:
    text = """
    <table><tr><td>HTML</td></tr></table>

    | Markdown |
    | --- |
    | fallback |
    """

    tables = parse_table_text(text)

    assert len(tables) == 1
    assert tables[0].source == "html"
    assert tables[0].rows == [["HTML"]]


def test_markdown_table_is_used_when_html_is_absent() -> None:
    markdown = """
    | Material | Refs. |
    | --- | --- |
    | Al2O3 | 90 |
    | SiO2 | 91 |
    """

    rows = markdown_table_to_rows(markdown)
    tables = parse_table_text(markdown)

    assert rows == [
        ["Material", "Refs."],
        ["Al2O3", "90"],
        ["SiO2", "91"],
    ]
    assert len(tables) == 1
    assert tables[0].source == "markdown"
    assert tables[0].rows == rows


def test_no_table_returns_empty_list() -> None:
    assert parse_table_text("plain OCR text only") == []


def test_extract_markdown_text_keeps_legacy_mapping_fallbacks() -> None:
    assert extract_markdown_text({"text": "  table output  "}) == "table output"
