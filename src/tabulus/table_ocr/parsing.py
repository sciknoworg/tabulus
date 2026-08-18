from __future__ import annotations

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


class _SimpleTableHTMLParser(HTMLParser):
    """Legacy-compatible HTML table parser."""

    def __init__(self) -> None:
        super().__init__()
        self.in_tr = False
        self.in_td = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

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

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in ("td", "th") and self.in_td:
            self.in_td = False
            cell_text = "".join(self.current_cell).strip()
            cell_text = re.sub(r"\s+", " ", cell_text)
            self.current_row.append(cell_text)
            self.current_cell = []
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


def html_table_to_rows(html: str) -> list[list[str]]:
    """Convert one HTML table into the legacy Tabulus row representation."""

    parser = _SimpleTableHTMLParser()
    parser.feed(html)
    return _rectangularize(parser.rows)


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
