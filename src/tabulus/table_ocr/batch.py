from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tabulus.table_crops import TABLES_INDEX_NAME
from tabulus.table_ocr.base import (
    TableOCRAdapter,
    TableOCRInput,
    TableOCRStatus,
)
from tabulus.table_ocr.output import (
    parse_result_tables,
    write_table_ocr_artifacts,
)


BATCH_SUMMARY_NAME = "batch_summary.json"

_OWNED_ARTIFACT_NAMES = (
    "native",
    "parsed",
    "predictions",
    BATCH_SUMMARY_NAME,
)


def _clear_owned_reconstruction_artifacts(output_dir: Path) -> None:
    """Remove only Tabulus-owned artifacts from a previous batch run."""

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            f"Table reconstruction output is not a directory: {output_dir}"
        )

    for name in _OWNED_ARTIFACT_NAMES:
        path = output_dir / name

        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


@dataclass(frozen=True)
class TableOCRBatchItem:
    """Persisted outcome for one canonical MinerU table crop."""

    table_id: int
    source_image: str
    status: TableOCRStatus
    elapsed_seconds: float
    parsed_tables: int
    native_result: str
    parsed_result: str
    prediction_csv: str | None
    error: str | None = None


@dataclass(frozen=True)
class TableOCRBatchResult:
    """Summary of reconstructing every table crop through one adapter."""

    adapter_name: str
    display_name: str
    crop_root: str
    output_dir: str
    tables_requested: int
    tables_ok: int
    tables_empty: int
    tables_error: int
    prediction_csvs: int
    elapsed_seconds: float
    items: list[TableOCRBatchItem]
    summary_path: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["summary_path"] = str(self.summary_path)
        return data


def load_table_ocr_inputs(crop_root: Path) -> list[TableOCRInput]:
    """
    Load the canonical MinerU crop handoff in stable index order.

    The ``table_id`` values from ``tables_index.json`` are preserved exactly;
    the batch layer does not renumber or merge physical table crops.
    """

    crop_root = Path(crop_root)
    index_path = crop_root / TABLES_INDEX_NAME

    if not index_path.is_file():
        raise FileNotFoundError(
            f"Table-crop index not found: {index_path}"
        )

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Table-crop index is not valid JSON: {index_path}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"Table-crop index must contain a JSON object: {index_path}"
        )

    records = data.get("tables")

    if not isinstance(records, list):
        raise ValueError(
            f"Table-crop index has no valid 'tables' list: {index_path}"
        )

    inputs: list[TableOCRInput] = []
    seen_table_ids: set[int] = set()

    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                "Table-crop index contains a non-object record at "
                f"position {position}."
            )

        table = TableOCRInput.from_crop_record(crop_root, record)

        if table.table_id in seen_table_ids:
            raise ValueError(
                f"Duplicate table_id in table-crop index: {table.table_id}"
            )

        seen_table_ids.add(table.table_id)
        inputs.append(table)

    return inputs


def _relative_or_absolute(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _write_summary(
    summary_path: Path,
    payload: dict[str, Any],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_table_ocr_batch(
    crop_root: Path,
    output_dir: Path,
    adapter: TableOCRAdapter,
) -> TableOCRBatchResult:
    """
    Reconstruct every canonical table crop with one shared adapter instance.

    Reusing the same adapter object is deliberate: adapters such as
    PaddleOCR-VL can load their model pipeline once and reuse it across the
    complete crop set.

    Each physical MinerU crop remains an independent table throughout this
    stage. The batch layer writes native, parsed, and pre-resolution prediction
    artifacts for each crop but performs no reference processing, DOI
    enrichment, or continued-table merging.
    """

    crop_root = Path(crop_root)
    output_dir = Path(output_dir)

    tables = load_table_ocr_inputs(crop_root)
    capabilities = adapter.capabilities
    _clear_owned_reconstruction_artifacts(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[TableOCRBatchItem] = []
    batch_start = time.perf_counter()

    for table in tables:
        item_start = time.perf_counter()
        result = adapter.extract(table)
        elapsed = time.perf_counter() - item_start

        if result.table_id != table.table_id:
            raise ValueError(
                "Table OCR adapter changed table identity: "
                f"input table_id={table.table_id}, "
                f"result table_id={result.table_id}."
            )

        parsed_tables = parse_result_tables(result)
        artifacts = write_table_ocr_artifacts(
            result,
            output_dir,
            parsed_tables=parsed_tables,
        )

        items.append(
            TableOCRBatchItem(
                table_id=result.table_id,
                source_image=str(table.image_path),
                status=result.status,
                elapsed_seconds=elapsed,
                parsed_tables=len(parsed_tables),
                native_result=(
                    _relative_or_absolute(
                        artifacts.native_result,
                        output_dir,
                    )
                    or str(artifacts.native_result)
                ),
                parsed_result=(
                    _relative_or_absolute(
                        artifacts.parsed_result,
                        output_dir,
                    )
                    or str(artifacts.parsed_result)
                ),
                prediction_csv=_relative_or_absolute(
                    artifacts.prediction_csv,
                    output_dir,
                ),
                error=result.error,
            )
        )

    elapsed_seconds = time.perf_counter() - batch_start
    summary_path = output_dir / BATCH_SUMMARY_NAME

    batch_result = TableOCRBatchResult(
        adapter_name=capabilities.name,
        display_name=capabilities.display_name,
        crop_root=str(crop_root),
        output_dir=str(output_dir),
        tables_requested=len(tables),
        tables_ok=sum(item.status == "ok" for item in items),
        tables_empty=sum(item.status == "empty" for item in items),
        tables_error=sum(item.status == "error" for item in items),
        prediction_csvs=sum(
            item.prediction_csv is not None
            for item in items
        ),
        elapsed_seconds=elapsed_seconds,
        items=items,
        summary_path=summary_path,
    )

    _write_summary(summary_path, batch_result.to_dict())
    return batch_result
