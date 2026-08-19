from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tabulus.table_ocr.base import TableOCRResult
from tabulus.table_ocr.parsing import ParsedTable, parse_native_markdown


@dataclass(frozen=True)
class TableOCRArtifactPaths:
    """Files persisted for one table-reconstruction result."""

    native_result: Path
    parsed_result: Path
    prediction_csv: Path | None


def parse_result_tables(result: TableOCRResult) -> list[ParsedTable]:
    """
    Parse all table representations preserved by one OCR result.

    ``native_json`` and ``native_markdown`` are not treated as independent
    predictions. Parsing uses the preserved Markdown/HTML public view because
    that is the legacy-compatible source from which Tabulus reconstructs rows.
    """

    parsed: list[ParsedTable] = []

    for native_markdown in result.native_markdown:
        parsed.extend(parse_native_markdown(native_markdown))

    return parsed


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_prediction_csv(path: Path, table: ParsedTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(table.rows)


def write_table_ocr_artifacts(
    result: TableOCRResult,
    output_dir: Path,
    *,
    parsed_tables: Sequence[ParsedTable] | None = None,
) -> TableOCRArtifactPaths:
    """
    Persist the reconstruction boundary for one canonical table crop.

    The output directory contains three deliberately separate layers:

    ``native/``
        The complete adapter-neutral ``TableOCRResult`` including preserved
        adapter-native JSON/Markdown and provenance.

    ``parsed/``
        The legacy-compatible rectangular row/column reconstruction plus
        enough metadata to trace it back to the adapter and source crop.

    ``predictions/``
        A pre-reference-resolution CSV suitable for downstream processing and
        table-reconstruction evaluation. A prediction CSV is written only
        when exactly one table was parsed from the crop.

    ``parsed_tables`` can be supplied by another adapter-specific parser.
    When omitted, Tabulus uses the current legacy-compatible Markdown/HTML
    parser. This keeps persistence independent from Paddle-specific output
    while preserving a convenient default for the first implemented adapter.

    This function never performs reference resolution or DOI enrichment and
    therefore never writes a final resolved CSV.
    """

    output_dir = Path(output_dir)
    source_stem = Path(result.source_image).stem

    native_path = output_dir / "native" / f"{source_stem}.json"
    parsed_path = output_dir / "parsed" / f"{source_stem}.json"
    prediction_path = output_dir / "predictions" / f"{source_stem}.csv"

    _write_json(native_path, result.to_dict())

    if parsed_tables is None:
        parsed_tables = parse_result_tables(result)
    else:
        parsed_tables = list(parsed_tables)

    warnings: list[str] = []
    written_prediction: Path | None = None

    if result.status != "ok":
        warnings.append(
            f"OCR result status is {result.status!r}; no prediction CSV was written."
        )
    elif len(parsed_tables) == 0:
        warnings.append(
            "No structured table was parsed; no prediction CSV was written."
        )
    elif len(parsed_tables) > 1:
        warnings.append(
            "Multiple structured tables were parsed from one canonical crop; "
            "no prediction CSV was written because choosing one would be ambiguous."
        )
    else:
        _write_prediction_csv(prediction_path, parsed_tables[0])
        written_prediction = prediction_path

    parsed_payload = {
        "table_id": result.table_id,
        "adapter_name": result.adapter_name,
        "adapter_version": result.adapter_version,
        "model_version": result.model_version,
        "device": result.device,
        "source_image": str(result.source_image),
        "status": result.status,
        "tables_found": len(parsed_tables),
        "tables": [table.to_dict() for table in parsed_tables],
        "prediction_csv": (
            str(written_prediction.relative_to(output_dir))
            if written_prediction is not None
            else None
        ),
        "warnings": warnings,
    }
    _write_json(parsed_path, parsed_payload)

    return TableOCRArtifactPaths(
        native_result=native_path,
        parsed_result=parsed_path,
        prediction_csv=written_prediction,
    )
