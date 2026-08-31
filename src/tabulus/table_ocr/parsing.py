from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Literal, Mapping


TableParseSource = Literal["html", "markdown"]


@dataclass(frozen=True)
class ParsedTable:
    """Rectangular table reconstructed from a native OCR representation."""

    rows: list[list[str]]
    source: TableParseSource

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "rows": self.rows,
            "source": self.source,
        }


def extract_markdown_text(value: Any) -> str:
    """
    Return the textual Markdown/HTML representation from a native OCR value.

    PaddleOCR may expose its Markdown view either directly as a string or as
    a mapping containing a ``markdown_texts`` field. The additional keys are
    retained for compatibility with the legacy Tabulus Paddle service.
    """

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, Mapping):
        for key in ("markdown_texts", "markdown", "text", "md", "output"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        return json.dumps(value, ensure_ascii=False, indent=2)

    if value is None:
        return ""

    text = str(value)
    return text.strip()


def extract_html_tables(text: str) -> list[str]:
    """Extract complete HTML ``table`` elements from text."""

    return re.findall(r"(?is)<table\b.*?</table>", text)


@dataclass(frozen=True)
class _HTMLCell:
    text: str
    rowspan: int = 1
    colspan: int = 1


def _parse_span(value: str | None) -> int:
    """Return a positive HTML span, defaulting invalid values to one."""

    if value is None:
        return 1

    try:
        span = int(value)
    except (TypeError, ValueError):
        return 1

    return max(span, 1)


class _SimpleTableHTMLParser(HTMLParser):
    """HTML table parser that preserves row/column span metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.in_tr = False
        self.in_td = False
        self.current_cell: list[str] = []
        self.current_row: list[_HTMLCell] = []
        self.rows: list[list[_HTMLCell]] = []
        self.current_rowspan = 1
        self.current_colspan = 1

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()

        if tag == "tr":
            self.in_tr = True
            self.current_row = []
        elif tag in ("td", "th") and self.in_tr:
            self.in_td = True
            self.current_cell = []
            attributes = {
                name.lower(): value
                for name, value in attrs
            }
            self.current_rowspan = _parse_span(attributes.get("rowspan"))
            self.current_colspan = _parse_span(attributes.get("colspan"))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in ("td", "th") and self.in_td:
            self.in_td = False
            cell_text = "".join(self.current_cell).strip()
            cell_text = re.sub(r"\s+", " ", cell_text)
            self.current_row.append(
                _HTMLCell(
                    text=cell_text,
                    rowspan=self.current_rowspan,
                    colspan=self.current_colspan,
                )
            )
            self.current_cell = []
            self.current_rowspan = 1
            self.current_colspan = 1
        elif tag == "tr" and self.in_tr:
            self.in_tr = False
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []

    def handle_data(self, data: str) -> None:
        if self.in_td and data:
            self.current_cell.append(data)


def _rectangularize(rows: list[list[str]]) -> list[list[str]]:
    max_cols = max((len(row) for row in rows), default=0)
    return [
        row + [""] * (max_cols - len(row))
        for row in rows
    ]


def _expand_html_spans(rows: list[list[_HTMLCell]]) -> list[list[str]]:
    """
    Expand HTML row/column spans into a rectangular CSV-style grid.

    A merged cell keeps its value only at the upper-left grid position.
    Every other position covered by its rowspan/colspan becomes an empty
    placeholder. Rowspans that extend past the supplied HTML are clipped.
    """

    if not rows:
        return []

    grid: list[list[str | None]] = [[] for _ in rows]

    def ensure_width(row_index: int, width: int) -> None:
        if len(grid[row_index]) < width:
            grid[row_index].extend([None] * (width - len(grid[row_index])))

    for row_index, source_row in enumerate(rows):
        column_index = 0

        for cell in source_row:
            row_stop = min(len(rows), row_index + cell.rowspan)

            while True:
                required_width = column_index + cell.colspan

                for target_row in range(row_index, row_stop):
                    ensure_width(target_row, required_width)

                is_free = all(
                    grid[target_row][target_column] is None
                    for target_row in range(row_index, row_stop)
                    for target_column in range(column_index, required_width)
                )

                if is_free:
                    break

                column_index += 1

            for target_row in range(row_index, row_stop):
                for target_column in range(
                    column_index,
                    column_index + cell.colspan,
                ):
                    grid[target_row][target_column] = ""

            grid[row_index][column_index] = cell.text
            column_index += cell.colspan

    return _rectangularize(
        [
            [value if value is not None else "" for value in row]
            for row in grid
        ]
    )


def html_table_to_rows(html: str) -> list[list[str]]:
    """Convert one HTML table into the legacy Tabulus row representation."""

    parser = _SimpleTableHTMLParser()
    parser.feed(html)
    parser.close()
    return _expand_html_spans(parser.rows)


_OTSL_CELL_TOKENS = {
    "<fcel>",
    "<ecel>",
    "<lcel>",
    "<ucel>",
    "<xcel>",
}

_OTSL_TOKEN_PATTERN = re.compile(
    r"(<nl>|<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)"
)


def otsl_table_to_html(text: str) -> str:
    """
    Convert one OTSL table into deterministic HTML.

    Supported structural tokens are ``fcel``, ``ecel``, ``lcel``,
    ``ucel``, ``xcel``, and ``nl``.

    Cell text is preserved apart from HTML escaping. No semantic correction
    or content repair is performed. Short rows are padded with empty cells to
    the width of the widest OTSL row.
    """

    if not isinstance(text, str) or not text.strip():
        return ""

    parts = [
        part
        for part in _OTSL_TOKEN_PATTERN.split(text)
        if part
    ]

    rows: list[list[tuple[str, str]]] = []
    current_row: list[tuple[str, str]] = []
    index = 0

    while index < len(parts):
        part = parts[index]

        if part == "<nl>":
            if current_row:
                rows.append(current_row)
                current_row = []
            index += 1
            continue

        if part in _OTSL_CELL_TOKENS:
            cell_text = ""

            if (
                part == "<fcel>"
                and index + 1 < len(parts)
                and parts[index + 1] not in _OTSL_CELL_TOKENS
                and parts[index + 1] != "<nl>"
            ):
                cell_text = parts[index + 1]
                index += 1

            current_row.append((part, cell_text))

        index += 1

    if current_row:
        rows.append(current_row)

    if not rows:
        return ""

    num_cols = max(len(row) for row in rows)
    if num_cols == 0:
        return ""

    rows = [
        row + [("<ecel>", "")] * (num_cols - len(row))
        for row in rows
    ]

    def horizontal_span(row_index: int, col_index: int) -> int:
        span = 1
        next_col = col_index + 1

        while (
            next_col < num_cols
            and rows[row_index][next_col][0] in ("<lcel>", "<xcel>")
        ):
            span += 1
            next_col += 1

        return span

    def vertical_span(row_index: int, col_index: int) -> int:
        span = 1
        next_row = row_index + 1

        while (
            next_row < len(rows)
            and rows[next_row][col_index][0] in ("<ucel>", "<xcel>")
        ):
            span += 1
            next_row += 1

        return span

    output = ["<table>"]

    for row_index, row in enumerate(rows):
        output.append("<tr>")

        for col_index, (token, cell_text) in enumerate(row):
            if token in ("<lcel>", "<ucel>", "<xcel>"):
                continue

            if token not in ("<fcel>", "<ecel>"):
                continue

            rowspan = vertical_span(row_index, col_index)
            colspan = horizontal_span(row_index, col_index)

            attributes: list[str] = []

            if rowspan > 1:
                attributes.append(f'rowspan="{rowspan}"')

            if colspan > 1:
                attributes.append(f'colspan="{colspan}"')

            attribute_text = (
                " " + " ".join(attributes)
                if attributes
                else ""
            )

            escaped = html.escape(cell_text.strip(), quote=False)

            output.append(
                f"<td{attribute_text}>{escaped}</td>"
            )

        output.append("</tr>")

    output.append("</table>")
    return "".join(output)


def extract_markdown_tables(text: str) -> list[str]:
    """Extract GitHub-style pipe tables from Markdown text."""

    lines = text.splitlines()
    tables: list[str] = []

    def is_table_line(line: str) -> bool:
        stripped = line.strip()
        return "|" in stripped and len(stripped) >= 3

    def is_separator(line: str) -> bool:
        stripped = line.strip().replace(" ", "")
        return "---" in stripped and "|" in stripped

    index = 0

    while index < len(lines):
        if not is_table_line(lines[index]):
            index += 1
            continue

        buffer = [lines[index]]
        saw_separator = is_separator(lines[index])
        next_index = index + 1

        while next_index < len(lines) and is_table_line(lines[next_index]):
            buffer.append(lines[next_index])
            if is_separator(lines[next_index]):
                saw_separator = True
            next_index += 1

        if saw_separator and len(buffer) >= 2:
            tables.append("\n".join(buffer).strip() + "\n")

        index = next_index

    return tables


def markdown_table_to_rows(markdown_table: str) -> list[list[str]]:
    """Convert one GitHub-style Markdown table into rectangular rows."""

    rows = [
        row.strip()
        for row in markdown_table.splitlines()
        if row.strip()
    ]

    if len(rows) < 2:
        return []

    separator_index: int | None = None

    for index, row in enumerate(rows):
        if "|" in row and "---" in row:
            separator_index = index
            break

    if separator_index is None or separator_index == 0:
        return []

    def split_row(row: str) -> list[str]:
        parts = [cell.strip() for cell in row.split("|")]

        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]

        return parts

    header = split_row(rows[0])
    body = [
        split_row(row)
        for row in rows[separator_index + 1 :]
    ]

    return _rectangularize([header] + body)


def parse_table_text(text: str) -> list[ParsedTable]:
    """
    Parse tables using the legacy Tabulus preference order.

    HTML tables are preferred. Markdown pipe tables are used only when no
    HTML table is present.
    """

    html_tables = extract_html_tables(text)

    if html_tables:
        return [
            ParsedTable(
                rows=html_table_to_rows(html),
                source="html",
            )
            for html in html_tables
        ]

    return [
        ParsedTable(
            rows=markdown_table_to_rows(markdown),
            source="markdown",
        )
        for markdown in extract_markdown_tables(text)
        if markdown_table_to_rows(markdown)
    ]


def parse_native_markdown(value: Any) -> list[ParsedTable]:
    """Parse tables from one preserved native Markdown result."""

    return parse_table_text(extract_markdown_text(value))
