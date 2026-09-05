from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulus.bibliography.models import BibliographyEntry
from tabulus.reference_matching.matching import (
    NUMERIC_POSITION_METHOD,
    detect_reference_column,
    looks_like_reference_header,
    match_reference_value,
    normalize_text,
)


REFERENCE_MATCHES_NAME = "reference_matches.json"
REFERENCES_DIR_NAME = "references"


@dataclass(frozen=True)
class ReferenceMatchingResult:
    reference_tables_selected: int
    reference_tables_checked: int
    reference_tables_skipped: int
    matched_tables: tuple[dict[str, Any], ...]
    skipped_tables: tuple[dict[str, Any], ...]
    output_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "numeric_reference_semantics": (
                "1-based bibliography position in normalized GROBID TEI order"
            ),
            "reference_tables_selected": self.reference_tables_selected,
            "reference_tables_checked": self.reference_tables_checked,
            "reference_tables_skipped": self.reference_tables_skipped,
            "matched_tables": list(self.matched_tables),
            "skipped_tables": list(self.skipped_tables),
        }


class _SkippedParsedTable(Exception):
    def __init__(self, reason: str, parsed_table_count: int):
        self.reason = reason
        self.parsed_table_count = parsed_table_count
        super().__init__(reason)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _load_bibliography(path: Path) -> tuple[BibliographyEntry, ...]:
    payload = _load_json_object(path, label="Bibliography artifact")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Bibliography artifact must contain an entries list.")

    result: list[BibliographyEntry] = []
    seen_indices: set[int] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Bibliography entries must be JSON objects.")
        index = item.get("index")
        if not isinstance(index, int) or index <= 0 or index in seen_indices:
            raise ValueError("Bibliography entry indices must be unique positive integers.")
        seen_indices.add(index)
        result.append(
            BibliographyEntry(
                index=index,
                raw=str(item.get("raw") or ""),
                doi=str(item.get("doi") or ""),
                source=str(item.get("source") or payload.get("bibliography_source") or ""),
            )
        )
    return tuple(result)


def _resolve_reconstruction_dir(
    selected_path: Path,
    selected_payload: dict[str, Any],
) -> Path:
    value = selected_payload.get("reconstruction_dir")
    if isinstance(value, str) and value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = selected_path.parent / path
        return path
    return selected_path.parent


def _resolve_source_path(reconstruction_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = reconstruction_dir / path
    return path


def _load_one_parsed_table(path: Path, table_id: int) -> list[list[str]]:
    payload = _load_json_object(path, label="Parsed table artifact")
    parsed_table_id = payload.get("table_id")
    if parsed_table_id is not None and parsed_table_id != table_id:
        raise ValueError(
            f"Parsed table identity does not match selected table {table_id}: {path}"
        )
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError(
            f"Parsed table artifact must contain a tables list for table {table_id}: {path}"
        )
    if len(tables) == 0:
        raise _SkippedParsedTable(
            reason="no_parsed_table",
            parsed_table_count=0,
        )
    if len(tables) > 1:
        raise _SkippedParsedTable(
            reason="multiple_parsed_tables",
            parsed_table_count=len(tables),
        )
    table = tables[0]
    if not isinstance(table, dict) or not isinstance(table.get("rows"), list):
        raise ValueError(f"Parsed table rows are unavailable for table {table_id}: {path}")
    rows: list[list[str]] = []
    for row in table["rows"]:
        if not isinstance(row, list):
            raise ValueError(f"Parsed table rows must be lists for table {table_id}: {path}")
        rows.append([normalize_text(cell) for cell in row])
    return rows


def _match_table(
    table_item: dict[str, Any],
    reconstruction_dir: Path,
    bibliography: tuple[BibliographyEntry, ...],
) -> dict[str, Any]:
    table_id = table_item.get("table_id")
    if not isinstance(table_id, int):
        raise ValueError("Selected reference table is missing an integer table_id.")

    parsed_path = _resolve_source_path(
        reconstruction_dir,
        table_item.get("source_parsed"),
    )
    if parsed_path is None:
        raise ValueError(f"Selected reference table {table_id} has no source_parsed path.")

    rows = _load_one_parsed_table(parsed_path, table_id)
    reference_column = detect_reference_column(rows)
    row_matches: list[dict[str, Any]] = []
    matches_found = 0
    matches_total = 0

    if reference_column is not None:
        first_non_empty_row: int | None = None
        for row_index, row in enumerate(rows):
            value = normalize_text(
                row[reference_column] if reference_column < len(row) else ""
            )
            if value:
                first_non_empty_row = row_index
                break

        for row_index, row in enumerate(rows):
            value = normalize_text(
                row[reference_column] if reference_column < len(row) else ""
            )
            if not value:
                continue

            is_header = (
                row_index == first_non_empty_row
                and looks_like_reference_header(value)
            )
            value_match = match_reference_value(value, bibliography)

            if not is_header:
                matches_total += 1
                if value_match.found:
                    matches_found += 1

            row_matches.append(
                {
                    "row_index": row_index,
                    "value": value,
                    "found": value_match.found,
                    "matched_reference_indices": list(
                        value_match.matched_reference_indices
                    ),
                    "matched_references": [
                        candidate.reference.raw
                        for candidate in value_match.candidates
                    ],
                    "doi": [
                        candidate.reference.doi
                        for candidate in value_match.candidates
                    ],
                    "match_provenance": [
                        candidate.to_dict()
                        for candidate in value_match.candidates
                    ],
                    "tokens_total": len(value_match.tokens),
                    "tokens_matched": (
                        len(value_match.tokens) - len(value_match.unmatched_tokens)
                    ),
                    "unmatched_tokens": list(value_match.unmatched_tokens),
                    "is_header": is_header,
                }
            )

    source_prediction = table_item.get("source_prediction")
    source_file = (
        Path(source_prediction).name
        if isinstance(source_prediction, str) and source_prediction
        else parsed_path.name
    )

    return {
        "table_id": table_id,
        "source_file": source_file,
        "source_parsed": str(parsed_path),
        "source_prediction": source_prediction,
        "reference_column_index": reference_column,
        "matches_found": matches_found,
        "matches_total": matches_total,
        "matches": row_matches,
    }


def match_selected_reference_tables(
    selected_reference_tables_path: Path,
    bibliography_path: Path,
    *,
    output_path: Path | None = None,
) -> ReferenceMatchingResult:
    """Match Stage 3 selected tables against the Stage 4 bibliography artifact.

    The function is deterministic and offline. It never calls Crossref or any
    metadata service and never mutates reconstruction prediction CSVs.
    """

    selected_path = Path(selected_reference_tables_path).expanduser()
    selected_payload = _load_json_object(
        selected_path,
        label="Selected reference tables manifest",
    )
    bibliography = _load_bibliography(Path(bibliography_path))
    reconstruction_dir = _resolve_reconstruction_dir(
        selected_path,
        selected_payload,
    )

    tables = selected_payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError("Selected reference tables manifest must contain a tables list.")

    if any(not isinstance(table, dict) for table in tables):
        raise ValueError("Selected reference table entries must be JSON objects.")

    matched_tables_list: list[dict[str, Any]] = []
    skipped_tables_list: list[dict[str, Any]] = []

    for table in tables:
        try:
            matched_tables_list.append(
                _match_table(table, reconstruction_dir, bibliography)
            )
        except _SkippedParsedTable as skipped:
            skipped_tables_list.append(
                {
                    "table_id": table.get("table_id"),
                    "source_status": table.get("source_status"),
                    "source_parsed": table.get("source_parsed"),
                    "source_prediction": table.get("source_prediction"),
                    "reason": skipped.reason,
                    "parsed_table_count": skipped.parsed_table_count,
                }
            )

    matched_tables = tuple(matched_tables_list)
    skipped_tables = tuple(skipped_tables_list)

    final_output_path = (
        Path(output_path).expanduser()
        if output_path is not None
        else reconstruction_dir / REFERENCES_DIR_NAME / REFERENCE_MATCHES_NAME
    )
    result = ReferenceMatchingResult(
        reference_tables_selected=len(tables),
        reference_tables_checked=len(matched_tables),
        reference_tables_skipped=len(skipped_tables),
        matched_tables=matched_tables,
        skipped_tables=skipped_tables,
        output_path=final_output_path,
    )

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
