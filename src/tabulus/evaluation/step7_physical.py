"""Exact and RMS evaluation of physical Step 7 table grids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from tabulus.evaluation.batch import content_hash, evaluate_pairs
from tabulus.evaluation.grid_inputs import FragmentPair, GridInput, load_grid
from tabulus.evaluation.grid_scoring import aggregate_grid_results
from tabulus.evaluation.rms import DEFAULT_NUMBER_THRESHOLD, DEFAULT_TEXT_THRESHOLD
from tabulus.resolved_export import ENRICHMENT_COLUMNS


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
    return load_grid(GridInput(path, tuple(columns), header))


def prepare_step7_pairs(
    prediction_path: Path, gold_path: Path | None = None, *,
    alignment_path: Path | None = None, canonical_gold_root: Path | None = None,
    fragment_keys: Sequence[str] | None = None,
) -> tuple[list[FragmentPair], dict]:
    """Prepare explicit physical pairs using enriched or canonical Step 2 gold."""
    if (gold_path is None) == (canonical_gold_root is None):
        raise ValueError("Supply enriched gold or a canonical Step 2 gold root.")
    prediction_path = Path(prediction_path).expanduser().resolve()
    prediction = _load(prediction_path)
    pred_columns = _metadata_columns(prediction, "enrichment_columns")
    if canonical_gold_root is not None:
        root = Path(canonical_gold_root).expanduser().resolve()
        gold_path = root / "tables_index.json"
        raw = json.loads(gold_path.read_text(encoding="utf-8-sig"))
        gold = {"tables": [
            {
                "table_key": t.get("benchmark_table_key", t.get("table_key")),
                "gold_csv": "tables/" + str(t.get("benchmark_table_key", t.get("table_key"))) + "/gold.csv",
            } for t in raw["tables"]
        ]}
        gold_columns = []
    else:
        gold_path = Path(gold_path).expanduser().resolve()
        gold = _load(gold_path)
        gold_columns = _metadata_columns(gold, "quality_columns")
    gold_tables = {}
    for table in gold["tables"]:
        key = table["table_key"]
        if not isinstance(key, str) or not key or key in gold_tables:
            raise ValueError(f"Invalid or duplicate gold table_key: {key!r}.")
        gold_tables[key] = table
    if not gold_tables:
        raise ValueError("The physical Step 7 gold manifest contains no tables.")
    selected = set(fragment_keys) if fragment_keys is not None else set(gold_tables)
    if not selected or selected - gold_tables.keys():
        raise ValueError("Unknown or empty fragment-key selection.")
    inputs = {prediction_path, gold_path}
    alignment = {}
    if alignment_path is not None:
        alignment_path = Path(alignment_path).expanduser().resolve()
        inputs.add(alignment_path)
        for table in _load(alignment_path)["tables"]:
            table_id, key = table["table_id"], table["table_key"]
            if type(table_id) is not int or table_id in alignment:
                raise ValueError(f"Invalid or duplicate alignment table_id: {table_id!r}.")
            if key not in gold_tables or key in alignment.values():
                raise ValueError(f"Invalid or duplicate alignment table_key: {key!r}.")
            alignment[table_id] = key
    paired, seen_ids = {}, set()
    for table in prediction["tables"]:
        table_id = table["table_id"]
        if type(table_id) is not int or table_id in seen_ids:
            raise ValueError(f"Invalid or duplicate prediction table_id: {table_id!r}.")
        seen_ids.add(table_id)
        key = alignment.get(table_id) if alignment_path is not None else table.get("table_key")
        if key not in gold_tables:
            raise ValueError(f"Prediction table {table_id} requires an explicit mapping to a gold table_key.")
        if key in paired:
            raise ValueError(f"Multiple predictions map to gold fragment {key}.")
        paired[key] = table
    pairs = []
    source_provenance = {str(path): content_hash(path) for path in sorted(inputs)}
    for key, expected in gold_tables.items():
        if key not in selected:
            continue
        observed = paired.get(key)
        source = GridInput(
            _csv_path(gold_path, expected["gold_csv"]),
            tuple(gold_columns), expected.get("header_row_index"),
        )
        pred = (
            GridInput(_csv_path(prediction_path, observed["resolved_csv"]),
                      tuple(pred_columns), observed.get("header_row_index"))
            if observed is not None else None
        )
        pairs.append(FragmentPair(
            key, source, pred, group="physical-step7",
            provenance={
                "table_key": key, "prediction_table_id": observed["table_id"] if observed else None,
                "preparation_inputs": source_provenance,
            },
        ))
    return pairs, {"excluded_columns": {"gold": gold_columns, "prediction": pred_columns}, "inputs": inputs}


def evaluate_step7_physical(
    prediction_path: Path, gold_path: Path | None = None, *,
    alignment_path: Path | None = None, metric: str = "both",
    metrics: Sequence[str] | None = None, canonical_gold_root: Path | None = None,
    fragment_keys: Sequence[str] | None = None,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    number_threshold: float = DEFAULT_NUMBER_THRESHOLD,
    workers: int = 1, checkpoint_dir: Path | None = None, resume: bool = False,
) -> dict[str, Any]:
    """Evaluate physical grids through the shared core; 'both' retains its meaning."""
    if metric not in {"both", "exact", "rms", "exact-mapping"}:
        raise ValueError(f"Unsupported physical Step 7 metric: {metric!r}.")
    selected = metrics if metrics is not None else (("exact", "rms") if metric == "both" else (metric,))
    pairs, info = prepare_step7_pairs(
        prediction_path, gold_path, alignment_path=alignment_path,
        canonical_gold_root=canonical_gold_root, fragment_keys=fragment_keys,
    )
    batch = evaluate_pairs(
        pairs, metrics=selected, text_threshold=text_threshold, number_threshold=number_threshold,
        workers=workers, output_dir=checkpoint_dir, resume=resume,
    )
    results = batch["results"]
    aggregate = aggregate_grid_results(results)
    inputs = set(info["inputs"])
    for pair in pairs:
        inputs.add(pair.gold.path)
        if pair.prediction is not None:
            inputs.add(pair.prediction.path)
    return {
        "schema_version": 1,
        "scope": "Original physical table grid including headers and citation text.",
        "excluded_columns": info["excluded_columns"], "score_scale": "[0,100]",
        "normalization": {
            "exact": "Unicode NFC and collapsed whitespace, with case and punctuation preserved.",
            "rms": "Step 2 RMS CSV normalization and original/transposed prediction selection.",
            "exact-mapping": "Step 2 normalized mapping keys and exact string values.",
        },
        "text_threshold": text_threshold, "number_threshold": number_threshold,
        "aggregation": "Unweighted per-gold-fragment macro mean. Missing and empty exports score zero.",
        "coverage": {
            "expected": len(results), "exported": sum(r["status"] == "scored" for r in results),
            "missing": sum(r["status"] == "missing" for r in results),
            "empty": sum(r["status"] == "empty" for r in results),
            **({"unusable": sum(r["status"] == "unusable" for r in results)}
               if any(r["status"] == "unusable" for r in results) else {}),
        },
        **{key: aggregate[key] for key in ("macro", "micro", "pooled_counts", "population")},
        "exact_grid_matches": (
            sum(row["statistics"]["exact"]["exact_grid"] for row in results) if "exact" in selected else None
        ),
        "inputs": [{"path": str(path), "sha256": content_hash(path)} for path in sorted(inputs)],
        "tables": [
            {
                "table_key": row["fragment_id"],
                "prediction_table_id": row["provenance"]["prediction_table_id"],
                "gold_csv": row["inputs"]["gold"]["path"],
                "prediction_csv": row["inputs"]["prediction"]["path"] if row["inputs"]["prediction"] else None,
                **{key: row[key] for key in ("status", "metrics", "statistics", "population")},
                **({"exact_cells": row["exact_cells"]} if "exact_cells" in row else {}),
            } for row in results
        ],
    }
