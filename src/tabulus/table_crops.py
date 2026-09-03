from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulus.mineru.tables import (
    discover_tables,
    find_content_list,
    load_content_items,
)
from tabulus.models import TableRegion


TABLES_INDEX_NAME = "tables_index.json"


@dataclass
class TableCropExportResult:
    """Result of exporting table crops into a stable handoff directory."""

    tables_found: int
    crops_saved: int
    refs_start_page: int | None
    output_dir: Path
    images_dir: Path
    index_path: Path
    tables: list[dict[str, Any]]
    image_format_policy: str = "preserve-source-extension"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables_found": self.tables_found,
            "crops_saved": self.crops_saved,
            "refs_start_page": self.refs_start_page,
            "image_format_policy": self.image_format_policy,
            "images_dir": str(self.images_dir),
            "tables": self.tables,
        }


def _table_image_name(table: TableRegion) -> str:
    suffix = table.source_image_path.suffix.lower() or ".png"
    page_nr = table.page_nr or 0

    return f"page_{page_nr:03d}_table_{table.table_id:03d}{suffix}"


def _table_record(
    table: TableRegion,
    image_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    relative_image = image_path.relative_to(output_dir).as_posix()

    return {
        "table_id": table.table_id,
        "page_nr": table.page_nr,
        "in_references": table.in_references,
        "image": relative_image,
        "image_name": image_path.name,
        "bbox": table.bbox,
        "table_caption": table.caption,
        "table_footnote": table.footnote,
        "mineru_img_path": table.mineru_img_path,
        "mineru_source_image": str(table.source_image_path),
        "mineru_table_body": table.mineru_table_body,
        "source": "mineru",
    }


def export_table_crops(
    tables: list[TableRegion],
    output_dir: Path,
    *,
    refs_start_page: int | None = None,
) -> TableCropExportResult:
    """Copy table crop images and write the normalized tables_index.json."""

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for table in tables:
        if not table.source_image_path.is_file():
            raise FileNotFoundError(
                f"Table {table.table_id} image not found: "
                f"{table.source_image_path}"
            )

        destination = images_dir / _table_image_name(table)
        shutil.copyfile(table.source_image_path, destination)
        records.append(
            _table_record(
                table=table,
                image_path=destination,
                output_dir=output_dir,
            )
        )

    result = TableCropExportResult(
        tables_found=len(tables),
        crops_saved=len(records),
        refs_start_page=refs_start_page,
        output_dir=output_dir,
        images_dir=images_dir,
        index_path=output_dir / TABLES_INDEX_NAME,
        tables=records,
    )

    result.index_path.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


def export_mineru_table_crops(
    mineru_output_dir: Path,
    output_dir: Path,
) -> TableCropExportResult:
    """Export table crops from an existing MinerU output directory."""

    content_list = find_content_list(mineru_output_dir)
    items = load_content_items(content_list)
    table_items = [item for item in items if item.get("type") == "table"]
    unmaterializable_tables: list[dict[str, Any]] = []

    for table_id, item in enumerate(table_items, start=1):
        img_path = item.get("img_path")
        if isinstance(img_path, str) and img_path.strip():
            continue
        page_idx = item.get("page_idx")
        unmaterializable_tables.append(
            {
                "table_id": table_id,
                "page_nr": page_idx + 1 if isinstance(page_idx, int) else None,
                "bbox": item.get("bbox"),
                "reason": "missing_img_path",
            }
        )

    tables, refs_start_page = discover_tables(
        mineru_output_dir,
        skip_missing_img_path=True,
    )
    result = export_table_crops(
        tables=tables,
        output_dir=output_dir,
        refs_start_page=refs_start_page,
    )

    result.tables_found = len(table_items)

    data = result.to_dict()
    data["unmaterializable_count"] = len(unmaterializable_tables)
    data["unmaterializable_tables"] = unmaterializable_tables
    data["mineru_output_dir"] = str(Path(mineru_output_dir))
    data["mineru_content_list"] = str(content_list)

    result.index_path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result
