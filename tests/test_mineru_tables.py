import json
from pathlib import Path

import pytest

from tabulus.mineru.tables import (
    discover_tables,
    find_content_list,
    find_references_start_page,
)


def make_mineru_output(tmp_path: Path) -> Path:
    output = tmp_path / "paper" / "hybrid_auto"
    images = output / "images"
    images.mkdir(parents=True)

    (images / "table1.jpg").write_bytes(b"table-1")
    (images / "table2.jpg").write_bytes(b"table-2")

    content = [
        {
            "type": "text",
            "text": "Introduction",
            "page_idx": 0,
        },
        {
            "type": "table",
            "page_idx": 5,
            "img_path": "images/table1.jpg",
            "bbox": [10, 20, 30, 40],
            "table_caption": ["TABLE I. Example table."],
            "table_footnote": ["Example footnote."],
            "table_body": "<table><tr><td>A</td></tr></table>",
        },
        {
            "type": "title",
            "text": "References",
            "page_idx": 9,
        },
        {
            "type": "table",
            "page_idx": 10,
            "img_path": "images/table2.jpg",
            "bbox": [50, 60, 70, 80],
            "table_caption": ["TABLE II. Reference table."],
            "table_footnote": [],
            "table_body": "<table><tr><td>B</td></tr></table>",
        },
    ]

    path = output / "paper_content_list.json"
    path.write_text(
        json.dumps(content),
        encoding="utf-8",
    )

    return tmp_path / "paper"


def test_find_content_list(tmp_path):
    root = make_mineru_output(tmp_path)

    result = find_content_list(root)

    assert result.name == "paper_content_list.json"


def test_discover_tables(tmp_path):
    root = make_mineru_output(tmp_path)

    tables, refs_start_page = discover_tables(root)

    assert len(tables) == 2
    assert refs_start_page == 10

    first = tables[0]

    assert first.table_id == 1
    assert first.page_nr == 6
    assert first.bbox == [10, 20, 30, 40]
    assert first.caption == ["TABLE I. Example table."]
    assert first.footnote == ["Example footnote."]
    assert first.mineru_table_body == (
        "<table><tr><td>A</td></tr></table>"
    )
    assert first.in_references is False
    assert first.source_image_path.name == "table1.jpg"

    second = tables[1]

    assert second.page_nr == 11
    assert second.in_references is True


def test_references_start_page_uses_last_heading():
    items = [
        {
            "type": "title",
            "text": "References",
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "Main article",
            "page_idx": 2,
        },
        {
            "type": "title",
            "text": "References",
            "page_idx": 8,
        },
    ]

    assert find_references_start_page(items) == 9


def test_missing_content_list_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_content_list(tmp_path)
