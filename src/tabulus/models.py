from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TableRegion:
    """A table region detected in a scientific document."""

    table_id: int
    page_nr: int | None
    image_path: Path
    source_image_path: Path
    mineru_img_path: str | None = None
    bbox: list[float] | None = None
    caption: list[str] = field(default_factory=list)
    footnote: list[str] = field(default_factory=list)
    mineru_table_body: str | None = None
    in_references: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        data["source_image_path"] = str(self.source_image_path)
        return data


@dataclass
class MinerUTableExtractionResult:
    """Result of extracting table regions with MinerU."""

    tables_found: int
    crops_saved: int
    refs_start_page: int | None
    tables: list[TableRegion]
    mineru_output_dir: Path
    content_list_path: Path
    duration_seconds: float
    mineru_version: str | None = None
    backend: str = "hybrid-engine"
    effort: str = "high"
    method: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables_found": self.tables_found,
            "crops_saved": self.crops_saved,
            "refs_start_page": self.refs_start_page,
            "tables": [table.to_dict() for table in self.tables],
            "mineru_output_dir": str(self.mineru_output_dir),
            "content_list_path": str(self.content_list_path),
            "duration_seconds": self.duration_seconds,
            "mineru_version": self.mineru_version,
            "backend": self.backend,
            "effort": self.effort,
            "method": self.method,
        }
