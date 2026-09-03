import json
from pathlib import Path

import pytest

from tabulus.models import TableRegion
from tabulus.table_crops import (
    TABLES_INDEX_NAME,
    export_mineru_table_crops,
    export_table_crops,
)


def make_mineru_output(tmp_path: Path) -> Path:
    output = tmp_path / "paper" / "hybrid_auto"
    images = output / "images"
    images.mkdir(parents=True)

    (images / "table1.jpg").write_bytes(b"table-1")
    (images / "table2.png").write_bytes(b"table-2")

    content = [
        {
            "type": "table",
            "page_idx": 2,
            "img_path": "images/table1.jpg",
            "bbox": [10, 20, 30, 40],
            "table_caption": ["TABLE I. Example table."],
            "table_footnote": ["Example footnote."],
            "table_body": "<table><tr><td>A</td></tr></table>",
        },
        {
            "type": "text",
            "text": "Not a table",
            "page_idx": 3,
            "img_path": "images/not-a-table.jpg",
        },
        {
            "type": "title",
            "text": "References",
            "page_idx": 8,
        },
        {
            "type": "table",
            "page_idx": 9,
            "img_path": "images/table2.png",
            "bbox": [50, 60, 70, 80],
            "table_caption": [],
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


def test_export_mineru_table_crops_writes_index_and_copies_images(tmp_path):
    mineru_root = make_mineru_output(tmp_path)
    out_dir = tmp_path / "table_crops"

    result = export_mineru_table_crops(
        mineru_output_dir=mineru_root,
        output_dir=out_dir,
    )

    index_path = out_dir / TABLES_INDEX_NAME
    data = json.loads(index_path.read_text(encoding="utf-8"))

    assert result.index_path == index_path
    assert data["tables_found"] == 2
    assert data["crops_saved"] == 2
    assert data["refs_start_page"] == 9
    assert data["image_format_policy"] == "preserve-source-extension"
    assert data["mineru_output_dir"] == str(mineru_root)
    assert data["mineru_content_list"].endswith("paper_content_list.json")

    first = data["tables"][0]

    assert first["table_id"] == 1
    assert first["page_nr"] == 3
    assert first["in_references"] is False
    assert first["image"] == "images/page_003_table_001.jpg"
    assert first["image_name"] == "page_003_table_001.jpg"
    assert first["bbox"] == [10, 20, 30, 40]
    assert first["table_caption"] == ["TABLE I. Example table."]
    assert first["table_footnote"] == ["Example footnote."]
    assert first["mineru_img_path"] == "images/table1.jpg"
    assert first["mineru_table_body"] == (
        "<table><tr><td>A</td></tr></table>"
    )
    assert first["source"] == "mineru"

    second = data["tables"][1]

    assert second["in_references"] is True
    assert second["image"] == "images/page_010_table_002.png"

    assert (out_dir / first["image"]).read_bytes() == b"table-1"
    assert (out_dir / second["image"]).read_bytes() == b"table-2"
    assert len(list((out_dir / "images").iterdir())) == 2


def test_export_table_crops_raises_for_missing_source_image(tmp_path):
    table = TableRegion(
        table_id=1,
        page_nr=1,
        image_path=tmp_path / "missing.jpg",
        source_image_path=tmp_path / "missing.jpg",
        mineru_img_path="images/missing.jpg",
    )

    with pytest.raises(FileNotFoundError):
        export_table_crops(
            tables=[table],
            output_dir=tmp_path / "table_crops",
        )


def test_export_mineru_table_crops_records_unmaterializable_tables(tmp_path):
    mineru_root = make_mineru_output(tmp_path)
    content_list = next(mineru_root.rglob("*_content_list.json"))
    content = json.loads(content_list.read_text(encoding="utf-8"))
    content.insert(
        1,
        {
            "type": "table",
            "page_idx": 4,
            "img_path": "",
            "bbox": [1, 2, 3, 4],
            "table_caption": [],
            "table_footnote": [],
        },
    )
    content_list.write_text(json.dumps(content), encoding="utf-8")

    out_dir = tmp_path / "table_crops"
    result = export_mineru_table_crops(
        mineru_output_dir=mineru_root,
        output_dir=out_dir,
    )
    data = json.loads(result.index_path.read_text(encoding="utf-8"))

    assert result.tables_found == 3
    assert result.crops_saved == 2
    assert data["tables_found"] == 3
    assert data["crops_saved"] == 2
    assert data["unmaterializable_count"] == 1
    assert data["unmaterializable_tables"] == [
        {
            "table_id": 2,
            "page_nr": 5,
            "bbox": [1, 2, 3, 4],
            "reason": "missing_img_path",
        }
    ]
    assert [table["table_id"] for table in data["tables"]] == [1, 3]
