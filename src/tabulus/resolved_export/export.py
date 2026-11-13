from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulus.reference_matching.matching import normalize_text
from tabulus.resolved_export.csv_io import read_csv
from tabulus.resolved_export.merge import (
    MERGED_ORIGIN_COLUMN,
    MERGED_TABLES_DIR_NAME,
    materialize_continuation_merges,
    plan_continuation_merges,
    resolve_tables_index_path,
)
from tabulus.resolved_export.orkg import (
    ORKG_DIR_NAME,
    ORKG_MANIFEST_NAME,
    StructurePlanner,
    write_orkg_projection,
)


RESOLVED_TABLES_DIR_NAME = "resolved_reference_tables"
RESOLVED_TABLES_MANIFEST_NAME = "resolved_tables.json"

FINAL_RESOLUTION_STATUSES = {
    "validated_with_doi",
    "validated_without_doi",
    "rejected",
}

ENRICHMENT_COLUMNS = (
    "tabulus_reference_indices",
    "tabulus_resolution_statuses",
    "tabulus_canonical_dois",
    "tabulus_canonical_titles",
    "tabulus_canonical_authors",
    "tabulus_canonical_years",
    "tabulus_canonical_venues",
    "tabulus_resolution_sources",
    "tabulus_resolution_confidences",
    "tabulus_resolution_reasons",
    "tabulus_raw_references",
    "tabulus_unmatched_tokens",
)


@dataclass(frozen=True)
class ResolvedCSVExportResult:
    """Result of deterministic Step 7 physical-table export."""

    reference_matches_path: Path
    reference_resolution_path: Path
    output_dir: Path
    manifest_path: Path
    tables_exported: int
    tables: tuple[dict[str, Any], ...]
    merge_continuations: bool = False
    continuation_groups: tuple[
        dict[str, Any],
        ...
    ] = ()
    orkg_manifest_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "reference_matches": str(
                self.reference_matches_path
            ),
            "reference_resolution": str(
                self.reference_resolution_path
            ),
            "output_dir": str(self.output_dir),
            "tables_exported": self.tables_exported,
            "enrichment_encoding": (
                "JSON arrays preserve positional correspondence "
                "across references in the same table cell"
            ),
            "enrichment_columns": list(
                ENRICHMENT_COLUMNS
            ),
            "merged_origin_column": (
                MERGED_ORIGIN_COLUMN
            ),
            "merge_continuations": (
                self.merge_continuations
            ),
            "tables": list(self.tables),
            "continuation_groups": list(
                self.continuation_groups
            ),
        }
        if self.orkg_manifest_path is not None:
            payload["orkg_manifest"] = str(self.orkg_manifest_path)
        return payload


def _load_json_object(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label} is not valid JSON: {path}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(
            f"{label} must contain a JSON object: {path}"
        )

    return value


def _resolution_registry(
    resolution_path: Path,
) -> dict[int, dict[str, Any]]:
    payload = _load_json_object(
        resolution_path,
        label="Step 6 reference-resolution artifact",
    )

    entries = payload.get("entries")

    if not isinstance(entries, list):
        raise ValueError(
            "Step 6 reference-resolution artifact must "
            "contain an entries list."
        )

    registry: dict[int, dict[str, Any]] = {}

    for position, entry in enumerate(
        entries,
        start=1,
    ):
        if not isinstance(entry, dict):
            raise ValueError(
                "Step 6 entry must be an object at "
                f"position {position}."
            )

        resolution = entry.get("resolution")

        if not isinstance(resolution, dict):
            raise ValueError(
                "Step 6 entry has no valid resolution "
                f"object at position {position}."
            )

        index = resolution.get("reference_index")

        if (
            not isinstance(index, int)
            or index <= 0
        ):
            raise ValueError(
                "Step 6 resolution has an invalid "
                f"reference_index at position {position}."
            )

        if index in registry:
            raise ValueError(
                "Duplicate Step 6 reference_index: "
                f"{index}"
            )

        status = resolution.get("status")

        if status not in FINAL_RESOLUTION_STATUSES:
            raise ValueError(
                "Step 6 artifact contains non-final "
                f"status {status!r} for reference "
                f"index {index}."
            )

        authors = resolution.get(
            "canonical_authors",
            [],
        )

        if not isinstance(authors, list):
            raise ValueError(
                "Step 6 canonical_authors must be a "
                f"list for reference index {index}."
            )

        registry[index] = resolution

    return registry


def _load_reference_matches(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json_object(
        path,
        label="Step 5 reference-matches artifact",
    )

    matched_tables = payload.get("matched_tables")

    if not isinstance(matched_tables, list):
        raise ValueError(
            "Step 5 reference-matches artifact must "
            "contain a matched_tables list."
        )

    if any(
        not isinstance(item, dict)
        for item in matched_tables
    ):
        raise ValueError(
            "Step 5 matched table entries must be objects."
        )

    return payload, matched_tables


def _reconstruction_dir(
    reference_matches_path: Path,
    matched_tables: list[dict[str, Any]],
) -> Path:
    path = Path(reference_matches_path).expanduser()

    # Prefer the actual Step 2 reconstruction provenance.
    roots = set()

    for table in matched_tables:
        value = table.get("source_parsed")
        if not isinstance(value, str) or not value:
            continue

        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            roots.add(candidate.parent.parent)

    if len(roots) > 1:
        raise ValueError(
            "Step 5 references multiple reconstruction directories: "
            + ", ".join(str(p) for p in sorted(roots))
        )

    if len(roots) == 1:
        return roots.pop()

    # Backward compatibility with the original Tabulus layout.
    if path.parent.name == "references":
        return path.parent.parent

    raise ValueError(
        "Could not determine reconstruction directory "
        "from Step 5 artifact provenance."
    )


def _prediction_path(
    reconstruction_dir: Path,
    table: dict[str, Any],
) -> Path:
    value = table.get("source_prediction")

    if not isinstance(value, str) or not value:
        raise ValueError(
            "Step 7 requires a source_prediction CSV "
            f"for table {table.get('table_id')!r}."
        )

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = reconstruction_dir / path

    if not path.is_file():
        raise FileNotFoundError(
            "Prediction CSV not found for table "
            f"{table.get('table_id')!r}: {path}"
        )

    return path


def _json_cell(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _row_enrichment(
    match: dict[str, Any],
    registry: dict[int, dict[str, Any]],
    *,
    allow_uncovered_identities: bool = False,
) -> list[str]:
    indices = match.get(
        "matched_reference_indices",
        [],
    )

    if not isinstance(indices, list):
        raise ValueError(
            "Step 5 matched_reference_indices must "
            "be a list."
        )

    raw_references = match.get("matched_references")
    if raw_references is not None and (
        not isinstance(raw_references, list) or len(raw_references) != len(indices)
    ):
        raise ValueError("Step 5 matched_references do not align with the bibliography indices.")
    if raw_references is None:
        raw_references = [""] * len(indices)
    resolutions: list[dict[str, Any]] = []

    for index, raw in zip(indices, raw_references):
        if (
            not isinstance(index, int)
            or index <= 0
        ):
            raise ValueError(
                "Step 5 matched reference indices "
                "must be positive integers."
            )

        resolution = registry.get(index)

        if resolution is None:
            if not allow_uncovered_identities:
                raise ValueError(
                    "Step 5 links bibliography index "
                    f"{index}, but Step 6 has no final "
                    "resolution for it."
                )
            resolution = {
                "status": "identity_unavailable",
                "raw_reference": raw,
                "reason": "No independently annotated identity for this bibliography index.",
            }
        recorded = normalize_text(raw)
        resolved = normalize_text(resolution.get("raw_reference"))
        if recorded and resolved and recorded != resolved:
            raise ValueError(f"Bibliography evidence differs between Steps 5 and 6 for index {index}.")

        resolutions.append(resolution)

    unmatched_tokens = match.get(
        "unmatched_tokens",
        [],
    )

    if not isinstance(unmatched_tokens, list):
        raise ValueError(
            "Step 5 unmatched_tokens must be a list."
        )

    return [
        _json_cell(indices),
        _json_cell(
            [
                item.get("status")
                for item in resolutions
            ]
        ),
        _json_cell(
            [
                str(
                    item.get(
                        "canonical_doi",
                        "",
                    )
                    or ""
                )
                for item in resolutions
            ]
        ),
        _json_cell(
            [
                str(
                    item.get(
                        "canonical_title",
                        "",
                    )
                    or ""
                )
                for item in resolutions
            ]
        ),
        _json_cell(
            [
                item.get(
                    "canonical_authors",
                    [],
                )
                for item in resolutions
            ]
        ),
        _json_cell(
            [
                item.get(
                    "canonical_year"
                )
                for item in resolutions
            ]
        ),
        _json_cell(
            [
                str(
                    item.get(
                        "canonical_venue",
                        "",
                    )
                    or ""
                )
                for item in resolutions
            ]
        ),
        _json_cell(
            [
                str(
                    item.get(
                        "source",
                        "",
                    )
                    or ""
                )
                for item in resolutions
            ]
        ),
        _json_cell(
            [
                item.get(
                    "confidence"
                )
                for item in resolutions
            ]
        ),
        _json_cell(
            [
                str(
                    item.get(
                        "reason",
                        "",
                    )
                    or ""
                )
                for item in resolutions
            ]
        ),
        _json_cell(
            [
                str(
                    item.get(
                        "raw_reference",
                        "",
                    )
                    or ""
                )
                for item in resolutions
            ]
        ),
        _json_cell(unmatched_tokens),
    ]


def _validate_and_plan_table(
    table: dict[str, Any],
    reconstruction_dir: Path,
    registry: dict[int, dict[str, Any]],
    output_dir: Path,
    *,
    allow_uncovered_identities: bool = False,
) -> dict[str, Any]:
    table_id = table.get("table_id")

    if not isinstance(table_id, int):
        raise ValueError(
            "Step 5 matched table is missing an "
            "integer table_id."
        )

    prediction_path = _prediction_path(
        reconstruction_dir,
        table,
    )
    rows = read_csv(prediction_path)
    reference_column = table.get("reference_column_index")
    if reference_column is not None and (
        type(reference_column) is not int
        or not 0 <= reference_column < max(map(len, rows), default=0)
    ):
        raise ValueError(f"Invalid reference-column index for table {table_id}: {reference_column!r}.")

    matches = table.get("matches")

    if not isinstance(matches, list):
        raise ValueError(
            f"Step 5 matches must be a list for table {table_id}."
        )

    by_row: dict[int, dict[str, Any]] = {}
    header_row_index: int | None = None

    for match in matches:
        if not isinstance(match, dict):
            raise ValueError(
                "Step 5 row match must be an object "
                f"for table {table_id}."
            )

        row_index = match.get("row_index")

        if (
            not isinstance(row_index, int)
            or row_index < 0
        ):
            raise ValueError(
                "Invalid Step 5 row_index for table "
                f"{table_id}: {row_index!r}"
            )

        if row_index >= len(rows):
            raise ValueError(
                "Step 5 row_index is outside the "
                "prediction CSV for table "
                f"{table_id}: {row_index} >= {len(rows)}"
            )

        if row_index in by_row:
            raise ValueError(
                "Duplicate Step 5 row_index "
                f"{row_index} for table {table_id}."
            )

        if reference_column is not None:
            if reference_column >= len(rows[row_index]):
                raise ValueError(f"Missing reference cell in table {table_id}, row {row_index}.")
            if "value" in match and normalize_text(match["value"]) != normalize_text(
                rows[row_index][reference_column]
            ):
                raise ValueError(f"Step 5 citation differs from table {table_id}, row {row_index}.")

        # Precompute this now so all Step 5 -> Step 6
        # references are validated before any output is written.
        enrichment = _row_enrichment(
            match,
            registry,
            allow_uncovered_identities=allow_uncovered_identities,
        )

        planned_match = dict(match)
        planned_match["_enrichment"] = enrichment
        by_row[row_index] = planned_match

        if match.get("is_header") is True:
            if (
                header_row_index is not None
                and header_row_index != row_index
            ):
                raise ValueError(
                    "Multiple Step 5 header rows found "
                    f"for table {table_id}."
                )

            header_row_index = row_index

    output_name = (
        f"{prediction_path.stem}_resolved.csv"
    )
    resolved_path = output_dir / output_name

    return {
        "table_id": table_id,
        "prediction_path": prediction_path,
        "rows": rows,
        "matches_by_row": by_row,
        "header_row_index": header_row_index,
        "reference_column_index": reference_column,
        "resolved_path": resolved_path,
    }


def _write_planned_table(
    plan: dict[str, Any],
) -> dict[str, Any]:
    table_id = plan["table_id"]
    rows = plan["rows"]
    by_row = plan["matches_by_row"]
    header_row_index = plan[
        "header_row_index"
    ]
    resolved_path = plan["resolved_path"]

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched_rows = 0
    resolution_occurrences = 0

    with resolved_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        for row_index, row in enumerate(rows):
            if row_index == header_row_index:
                appended = list(
                    ENRICHMENT_COLUMNS
                )
            else:
                match = by_row.get(row_index)

                if match is None:
                    appended = [
                        _json_cell([])
                        for _ in ENRICHMENT_COLUMNS
                    ]
                else:
                    matched_rows += 1
                    enrichment = match[
                        "_enrichment"
                    ]
                    appended = enrichment

                    indices = match.get(
                        "matched_reference_indices",
                        [],
                    )
                    resolution_occurrences += len(
                        indices
                    )

            writer.writerow(
                list(row) + appended
            )

    return {
        "table_id": table_id,
        "source_prediction": str(
            plan["prediction_path"]
        ),
        "resolved_csv": str(
            resolved_path
        ),
        "rows": len(rows),
        "reference_column_index": plan[
            "reference_column_index"
        ],
        "header_row_index": (
            header_row_index
        ),
        "matched_rows": matched_rows,
        "resolution_occurrences": (
            resolution_occurrences
        ),
        "merge_status": "physical_only",
    }


def export_resolved_csvs(
    reference_matches_path: Path,
    reference_resolution_path: Path,
    *,
    output_dir: Path | None = None,
    merge_continuations: bool = False,
    tables_index_path: Path | None = None,
    reconstruction_root: Path | None = None,
    orkg: bool = False,
    structure_planner: StructurePlanner | None = None,
    allow_uncovered_identities: bool = False,
) -> ResolvedCSVExportResult:
    """
    Deterministically export Step 7 resolved physical-table CSVs.

    Step 5 supplies row-level bibliography links. Step 6 supplies one
    final paper-level identity decision per bibliography index. This
    function joins the two without external lookup, model inference,
    or mutation of the Step 2 prediction CSVs.

    Physical table boundaries and row order are preserved. Logical
    continued-table merging is a separate optional Step 7 operation.
    """

    matches_path = Path(
        reference_matches_path
    ).expanduser()
    resolution_path = Path(
        reference_resolution_path
    ).expanduser()

    matches_payload, matched_tables = (
        _load_reference_matches(matches_path)
    )
    index_namespace = matches_payload.get("index_namespace", "bibliography_position")
    if not isinstance(index_namespace, str) or not index_namespace.strip():
        raise ValueError("Step 5 index_namespace must be a nonempty string.")
    if allow_uncovered_identities and index_namespace != "gold_index":
        raise ValueError("Uncovered identities are allowed only for controlled gold_index inputs.")
    registry = _resolution_registry(
        resolution_path
    )

    reconstruction_dir = (
        Path(reconstruction_root).expanduser()
        if reconstruction_root is not None
        else _reconstruction_dir(matches_path, matched_tables)
    )

    final_output_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else (
            reconstruction_dir
            / RESOLVED_TABLES_DIR_NAME
        )
    )

    # Validate the complete join before creating any output.
    plans = [
        _validate_and_plan_table(
            table,
            reconstruction_dir,
            registry,
            final_output_dir,
            allow_uncovered_identities=allow_uncovered_identities,
        )
        for table in matched_tables
    ]

    if len({plan["table_id"] for plan in plans}) != len(plans):
        raise ValueError("Duplicate Step 5 physical table IDs.")
    output_names = [plan["resolved_path"].name.casefold() for plan in plans]

    if len(output_names) != len(
        set(output_names)
    ):
        raise ValueError(
            "Step 7 resolved CSV filename collision "
            "detected across physical tables."
        )

    manifest_path = final_output_dir / RESOLVED_TABLES_MANIFEST_NAME
    input_paths = {path.resolve() for path in (matches_path, resolution_path)}
    input_paths.update(plan["prediction_path"].resolve() for plan in plans)
    index_path = Path(tables_index_path).expanduser() if tables_index_path is not None else None
    if index_path is not None:
        if (orkg or merge_continuations) and not index_path.is_file():
            raise FileNotFoundError(f"Step 1 tables_index.json not found: {index_path}")
        input_paths.add(index_path.resolve())
    output_paths = [manifest_path, *(plan["resolved_path"] for plan in plans)]
    if orkg:
        output_paths.append(final_output_dir / ORKG_DIR_NAME / ORKG_MANIFEST_NAME)
        output_paths.extend(
            final_output_dir / ORKG_DIR_NAME / f"{plan['prediction_path'].stem}_orkg.csv"
            for plan in plans
        )

    merge_plans: list[dict[str, Any]] = []

    if merge_continuations:
        resolved_tables_index_path = (
            resolve_tables_index_path(
                reconstruction_dir,
                tables_index_path,
            )
        )

        merge_plans = (
            plan_continuation_merges(
                resolved_tables_index_path,
                plans,
            )
        )
        output_paths.extend(
            final_output_dir / MERGED_TABLES_DIR_NAME / f"table_{plan['root_table_id']:03d}_merged_resolved.csv"
            for plan in merge_plans if plan["merge_status"] in {"ready", "ready_partial"}
        )
    if any(path.resolve() in input_paths for path in output_paths):
        raise ValueError("A Step 7 output path aliases an input artifact.")

    table_results = tuple(
        _write_planned_table(plan)
        for plan in plans
    )

    continuation_groups = (
        materialize_continuation_merges(
            merge_plans,
            plans,
            table_results,
            final_output_dir,
        )
        if merge_continuations
        else ()
    )

    orkg_manifest_path = (
        write_orkg_projection(
            plans=plans, registry=registry, reconstruction_dir=reconstruction_dir,
            output_dir=final_output_dir, planner=structure_planner,
            index_namespace=index_namespace,
            input_paths={"reference_matches": matches_path, "reference_resolution": resolution_path},
            tables_index_path=(
                index_path or reconstruction_dir.parent.parent / "tables_index.json"
            ),
        )
        if orkg else None
    )

    result = ResolvedCSVExportResult(
        reference_matches_path=matches_path,
        reference_resolution_path=resolution_path,
        output_dir=final_output_dir,
        manifest_path=manifest_path,
        tables_exported=len(
            table_results
        ),
        tables=table_results,
        merge_continuations=(
            merge_continuations
        ),
        continuation_groups=(
            continuation_groups
        ),
        orkg_manifest_path=orkg_manifest_path,
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.write_text(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return result
