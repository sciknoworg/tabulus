
"""Optional ORKG-compatible projection of Step 7 physical-table plans."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, Callable

from tabulus.reference_resolution.llm_client import OpenAICompatibleLLMError
from tabulus.resolved_export.structure import (
    StructureEvidence,
    direct_plan,
    execute_plan,
    native_structure,
    routing_reasons,
)
from tabulus.resolved_export.structural_llm import (
    OpenAICompatibleStructurePlanner,
    StructureConfigurationError,
    StructureResponseError,
)


ORKG_DIR_NAME = "orkg"
ORKG_MANIFEST_NAME = "manifest.json"
PAPER_COLUMNS = ("paper:doi", "paper:title", "paper:authors", "paper:publication_year")
StructurePlanner = Callable[[dict[str, Any]], tuple[Any, dict[str, Any]]]


def _structure_input(
    table: dict[str, Any], reconstruction_dir: Path, crop_records: dict[str, Any],
) -> tuple[str, StructureEvidence]:
    stem = table["prediction_path"].stem
    native_path = reconstruction_dir / "native" / f"{stem}.json"
    parsed_path = reconstruction_dir / "parsed" / f"{stem}.json"
    native = json.loads(native_path.read_text(encoding="utf-8")) if native_path.is_file() else {}
    if not isinstance(native, dict) or native.get("table_id", table["table_id"]) != table["table_id"]:
        raise ValueError("Native structure evidence does not identify the source table.")
    parsed = {}
    if not native.get("source_image") and parsed_path.is_file():
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict) or parsed.get("table_id", table["table_id"]) != table["table_id"]:
            raise ValueError("Parsed structure evidence does not identify the source table.")
    image_path = native.get("source_image") or parsed.get("source_image")
    key = PurePosixPath(image_path.replace("\\", "/")).stem if isinstance(image_path, str) else stem
    evidence = native_structure(table["rows"], native)
    crop = crop_records.get(key, {})
    body = crop.get("mineru_table_body")
    if isinstance(body, str) and body:
        mineru = native_structure(table["rows"], {"native_markdown": [body]})
        spans = tuple(
            {**span, "evidence_source": name}
            for name, value in (("adapter", evidence), ("mineru", mineru)) for span in value.spans
        )
        evidence = StructureEvidence(tuple(sorted(set(evidence.reasons + mineru.reasons))), spans)
    return key, evidence


def _paper_fields(resolution: dict[str, Any] | None) -> list[str] | None:
    if not resolution:
        return None
    status = resolution.get("status")
    doi = resolution.get("canonical_doi")
    if status == "validated_with_doi" and isinstance(doi, str) and doi.strip():
        return [doi.strip(), "", "", ""]
    title = resolution.get("canonical_title")
    authors = resolution.get("canonical_authors")
    year = resolution.get("canonical_year")
    if (
        status == "validated_without_doi" and isinstance(title, str) and title.strip()
        and isinstance(authors, list) and authors
        and all(isinstance(author, str) and author.strip() for author in authors)
        and type(year) is int and 1000 <= year <= 9999
    ):
        return ["", title, " and ".join(authors), str(year)]
    return None


def _load_crop_records(index_path: Path | None) -> dict[str, Any]:
    if index_path is None or not index_path.is_file():
        return {}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = {}
    for table in payload["tables"]:
        image_name = table.get("image_name")
        if isinstance(image_name, str):
            key = Path(image_name).stem
            if key in records:
                raise ValueError(f"Duplicate structural evidence for {key}.")
            records[key] = table
    return records


def write_orkg_projection(
    *,
    plans: list[dict[str, Any]],
    registry: dict[int, dict[str, Any]],
    reconstruction_dir: Path,
    output_dir: Path,
    tables_index_path: Path | None,
    planner: StructurePlanner | None,
    index_namespace: str,
    input_paths: dict[str, Path],
) -> Path:
    """Write projections and evidence. An unusable plan is a recorded abstention."""
    started = perf_counter()
    if planner is None:
        planner = OpenAICompatibleStructurePlanner()
    directory = output_dir / ORKG_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    crop_records = _load_crop_records(tables_index_path)
    tables = []
    for table in plans:
        table_started = perf_counter()
        path = table["prediction_path"]
        table_id = table["table_id"]
        item: dict[str, Any] = {
            "table_id": table_id, "table_key": path.stem, "source_prediction": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_rows": len(table["rows"]), "llm_invocations": 0,
            "unmatched_citations": [
                {"row": row, "column": table["reference_column_index"], "tokens": match["unmatched_tokens"]}
                for row, match in table["matches_by_row"].items()
                if not match.get("is_header") and match.get("unmatched_tokens")
            ],
        }
        try:
            key, evidence = _structure_input(table, reconstruction_dir, crop_records)
            item["table_key"] = key
            reasons = routing_reasons(table, evidence)
            item["route"] = "llm_assisted" if reasons else "deterministic"
            item["route_reasons"] = list(reasons)
            item["evidence_spans"] = list(evidence.spans)
            if reasons:
                model_input = {
                    "rows": table["rows"],
                    "reference_column": table["reference_column_index"],
                    "recorded_header": table["header_row_index"],
                    "explicit_citations": [
                        {"row": row, "occurrences": len(match["matched_reference_indices"])}
                        for row, match in table["matches_by_row"].items()
                        if match.get("matched_reference_indices")
                    ],
                    "routing_evidence": reasons,
                    "aligned_spans": evidence.spans,
                    "caption": crop_records.get(key, {}).get("table_caption"),
                }
                item["llm_invocations"] = 1
                proposal, metadata = planner(model_input)
                item["llm"] = metadata
            else:
                proposal = direct_plan(table)
            item["plan"] = proposal
            result = execute_plan(table, proposal, evidence)
        except (OpenAICompatibleLLMError, ValueError) as error:
            if isinstance(error, StructureConfigurationError):
                item["llm_invocations"] = 0
            if isinstance(error, StructureResponseError):
                item["llm"] = error.metadata
            item.update(status="abstained", error=str(error), runtime_seconds=perf_counter() - table_started)
            tables.append(item)
            continue

        item.update(result)
        headers = [f"{column['label']}<{column['datatype']}>" for column in result["columns"]]
        export_rows = []
        omissions = []
        for record_index, record in enumerate(result["records"]):
            values = [cell["value"] for cell in record["cells"]]
            if not record["links"]:
                omissions.append({"record": record_index, "reason": "no_linked_reference"})
            for link in record["links"]:
                resolution = registry.get(link["reference_index"])
                fields = _paper_fields(resolution)
                if resolution is not None:
                    link["identity"] = {
                        field: resolution.get(field)
                        for field in (
                            "status", "canonical_doi", "canonical_title", "canonical_authors",
                            "canonical_year", "canonical_venue",
                        )
                    }
                if fields is None:
                    status = (resolution or {}).get("status")
                    omissions.append({
                        "record": record_index, **link,
                        "reason": "identity_not_importable" if status else "missing_identity",
                    })
                    continue
                export_rows.append(values + fields)
                link["output_row"] = len(export_rows)

        csv_path = directory / f"{path.stem}_orkg.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers + list(PAPER_COLUMNS))
            writer.writerows(export_rows)
        item.update(
            status="exported" if export_rows else "no_importable_records",
            csv=csv_path.name, export_rows=len(export_rows), omissions=omissions,
            runtime_seconds=perf_counter() - table_started,
        )
        tables.append(item)

    manifest = directory / ORKG_MANIFEST_NAME
    if tables_index_path is not None and tables_index_path.is_file():
        input_paths = {**input_paths, "tables_index": tables_index_path}
    manifest.write_text(json.dumps({
        "schema_version": 1, "coordinate_system": "zero-based physical CSV rows and columns",
        "index_namespace": index_namespace,
        "inputs": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in input_paths.items()
        },
        "runtime_seconds": perf_counter() - started,
        "tables": tables,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
