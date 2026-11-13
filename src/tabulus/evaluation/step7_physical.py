"""Exact and RMS evaluation of physical Step 7 table grids."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from tabulus.evaluation.rms import (
    DEFAULT_NUMBER_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    relative_mapping_similarity,
    rows_to_rms_text,
)
from tabulus.evaluation.table_reconstruction import count_exact_cells
from tabulus.resolved_export import ENRICHMENT_COLUMNS
from tabulus.resolved_export.csv_io import read_csv


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Expected a schema_version 1 manifest in {path}.")
    return payload


def _csv_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def _metadata_columns(payload: dict[str, Any], field: str) -> list[str]:
    columns = payload.get(field)
    if (
        not isinstance(columns, list) or not columns
        or any(not isinstance(column, str) or column not in ENRICHMENT_COLUMNS for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError(f"Invalid Step 7 {field}.")
    return columns


def _original_grid(path: Path, columns: list[str], header: int | None) -> list[list[str]]:
    rows = read_csv(path)
    width = len(columns)
    if any(len(row) < width for row in rows):
        raise ValueError(f"A CSV row has fewer cells than its declared metadata columns in {path}.")
    if rows and header is not None:
        if type(header) is not int or header < 0 or header >= len(rows):
            raise ValueError(f"Invalid header_row_index in {path}.")
        if rows[header][-width:] != columns:
            raise ValueError(f"Declared metadata does not match the CSV header in {path}.")
    return [row[:-width] for row in rows]


def evaluate_step7_physical(
    prediction_path: Path,
    gold_path: Path,
    *,
    alignment_path: Path | None = None,
    metric: str = "both",
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    number_threshold: float = DEFAULT_NUMBER_THRESHOLD,
) -> dict[str, Any]:
    """Compare original grids using explicit fragment correspondence.

    Alignment JSON contains a tables list of table_id/table_key pairs. Without
    alignment, prediction entries must carry their own stable table_key.
    """
    if metric not in {"both", "exact", "rms"}:
        raise ValueError(f"Unsupported physical Step 7 metric: {metric!r}.")
    if not 0 <= text_threshold <= 1 or not 0 <= number_threshold <= 1:
        raise ValueError("RMS thresholds must be between zero and one.")
    prediction_path = Path(prediction_path).expanduser().resolve()
    gold_path = Path(gold_path).expanduser().resolve()
    prediction = _load(prediction_path)
    gold = _load(gold_path)
    pred_columns = _metadata_columns(prediction, "enrichment_columns")
    gold_columns = _metadata_columns(gold, "quality_columns")
    gold_tables = {}
    for table in gold["tables"]:
        key = table["table_key"]
        if not isinstance(key, str) or not key or key in gold_tables:
            raise ValueError(f"Invalid or duplicate gold table_key: {key!r}.")
        gold_tables[key] = table
    if not gold_tables:
        raise ValueError("The physical Step 7 gold manifest contains no tables.")

    inputs = {prediction_path, gold_path}
    alignment = {}
    if alignment_path is not None:
        alignment_path = Path(alignment_path).expanduser().resolve()
        inputs.add(alignment_path)
        for table in _load(alignment_path)["tables"]:
            table_id = table["table_id"]
            if type(table_id) is not int or table_id in alignment:
                raise ValueError(f"Invalid or duplicate alignment table_id: {table_id!r}.")
            key = table["table_key"]
            if key not in gold_tables or key in alignment.values():
                raise ValueError(f"Invalid or duplicate alignment table_key: {key!r}.")
            alignment[table_id] = key

    paired = {}
    seen_ids = set()
    for table in prediction["tables"]:
        table_id = table["table_id"]
        if type(table_id) is not int or table_id in seen_ids:
            raise ValueError(f"Invalid or duplicate prediction table_id: {table_id!r}.")
        seen_ids.add(table_id)
        key = alignment.get(table_id) if alignment_path is not None else table.get("table_key")
        if key not in gold_tables:
            raise ValueError(
                f"Prediction table {table_id} requires an explicit mapping to a gold table_key."
            )
        if key in paired:
            raise ValueError(f"Multiple predictions map to gold fragment {key}.")
        paired[key] = table
    metrics = ("exact", "rms") if metric == "both" else (metric,)
    results = []
    for key, expected in gold_tables.items():
        gold_csv = _csv_path(gold_path, expected["gold_csv"])
        inputs.add(gold_csv)
        gold_grid = _original_grid(gold_csv, gold_columns, expected.get("header_row_index"))
        observed = paired.get(key)
        pred_csv = _csv_path(prediction_path, observed["resolved_csv"]) if observed else None
        if pred_csv is not None:
            inputs.add(pred_csv)
        status = "missing"
        pred_grid = []
        if pred_csv is not None and pred_csv.is_file():
            pred_grid = _original_grid(pred_csv, pred_columns, observed.get("header_row_index"))
            status = "scored" if pred_grid else "empty"
        result = {
            "table_key": key,
            "prediction_table_id": observed["table_id"] if observed else None,
            "gold_csv": str(gold_csv),
            "prediction_csv": str(pred_csv) if pred_csv is not None else None,
            "status": status,
            "metrics": {},
        }
        if "exact" in metrics:
            counts = count_exact_cells(gold_grid, pred_grid)
            result["exact_cells"] = asdict(counts)
            result["metrics"]["exact"] = {
                "precision": 100 * counts.matching_cells / max(1, counts.predicted_nonempty_cells),
                "recall": 100 * counts.matching_cells / max(1, counts.gold_nonempty_cells),
                "f1": 200 * counts.matching_cells / max(
                    1, counts.gold_nonempty_cells + counts.predicted_nonempty_cells
                ),
            }
        if "rms" in metrics:
            scores = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            if status == "scored":
                scores = {
                    name: value * 100 for name, value in asdict(relative_mapping_similarity(
                        rows_to_rms_text(gold_grid), rows_to_rms_text(pred_grid),
                        text_threshold=text_threshold, number_threshold=number_threshold,
                    )).items()
                }
            result["metrics"]["rms"] = scores
        results.append(result)

    return {
        "schema_version": 1,
        "scope": "Original physical table grid including headers and citation text.",
        "excluded_columns": {"gold": gold_columns, "prediction": pred_columns},
        "score_scale": "[0,100]",
        "normalization": {
            "exact": "Unicode NFC and collapsed whitespace, with case and punctuation preserved.",
            "rms": "Step 2 RMS CSV normalization and original/transposed prediction selection.",
        },
        "text_threshold": text_threshold,
        "number_threshold": number_threshold,
        "aggregation": "Unweighted per-gold-fragment macro mean. Missing and empty exports score zero.",
        "coverage": {
            "expected": len(results),
            "exported": sum(table["status"] == "scored" for table in results),
            "missing": sum(table["status"] == "missing" for table in results),
            "empty": sum(table["status"] == "empty" for table in results),
        },
        "macro": {
            name: {
                field: mean(table["metrics"][name][field] for table in results)
                for field in ("precision", "recall", "f1")
            } for name in metrics
        },
        "exact_grid_matches": (
            sum(table["exact_cells"]["exact_grid"] and table["status"] == "scored" for table in results)
            if "exact" in metrics else None
        ),
        "inputs": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            } for path in sorted(inputs)
        ],
        "tables": results,
    }
