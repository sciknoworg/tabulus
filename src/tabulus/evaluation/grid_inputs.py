"""Input preparation for physical-grid evaluation, without benchmark assumptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from tabulus.resolved_export.csv_io import read_csv


@dataclass(frozen=True)
class GridInput:
    path: Path | None = None
    metadata_columns: tuple[str, ...] = ()
    header_row_index: int | None = None
    inline_grid: tuple[tuple[str, ...], ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "metadata_columns", tuple(self.metadata_columns))
        if self.inline_grid is not None:
            object.__setattr__(self, "inline_grid", tuple(tuple(row) for row in self.inline_grid))
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        if len(set(self.metadata_columns)) != len(self.metadata_columns):
            raise ValueError("Appended metadata columns must be distinct.")
        if self.path is None and self.inline_grid is None:
            raise ValueError("A grid input needs a CSV path or inline rows.")


@dataclass(frozen=True)
class FragmentPair:
    fragment_id: str
    gold: GridInput
    prediction: GridInput | None
    group: str = "default"
    prediction_status: str = "ok"
    provenance: dict = field(default_factory=dict)


def load_grid(source: GridInput) -> list[list[str]]:
    """Remove only explicitly declared appended columns on this input side."""
    rows = (
        [list(row) for row in source.inline_grid]
        if source.inline_grid is not None else read_csv(source.path)
    )
    columns = source.metadata_columns
    if not columns:
        return rows
    width = len(columns)
    if any(len(row) < width for row in rows):
        raise ValueError(f"A CSV row has fewer cells than its declared metadata columns in {source.path}.")
    header = source.header_row_index
    if rows and header is not None:
        if type(header) is not int or not 0 <= header < len(rows):
            raise ValueError(f"Invalid header_row_index in {source.path}.")
        if rows[header][-width:] != list(columns):
            raise ValueError(f"Declared metadata does not match the CSV header in {source.path}.")
    return [row[:-width] for row in rows]


def load_fragment_manifest(path: Path) -> list[FragmentPair]:
    """Read explicit pairs; relative CSV paths are relative to this manifest."""
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported fragment manifest schema.")

    def source(value):
        if value is None:
            return None
        csv_path = value.get("csv")
        if csv_path is not None:
            csv_path = Path(csv_path).expanduser()
            csv_path = csv_path if csv_path.is_absolute() else path.parent / csv_path
        rows = value.get("inline_grid")
        return GridInput(
            csv_path, tuple(value.get("metadata_columns", ())), value.get("header_row_index"),
            tuple(tuple(row) for row in rows) if rows is not None else None,
        )

    return [
        FragmentPair(
            fragment_id=item["fragment_id"], group=item.get("group", "default"),
            gold=source(item["gold"]), prediction=source(item.get("prediction")),
            prediction_status=item.get("prediction_status", "ok"),
            provenance={**item.get("provenance", {}), "fragment_manifest": str(path)},
        )
        for item in payload["pairs"]
    ]
