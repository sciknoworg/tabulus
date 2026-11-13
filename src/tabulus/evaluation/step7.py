
"""Evaluate Step 7 structure, scholarly evidence and projection independently."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from tabulus.evaluation.reference_resolution import _authors, _title_ok, _venue_ok
from tabulus.reference_resolution.scoring import normalize_doi
from tabulus.resolved_export.csv_io import read_csv
from tabulus.resolved_export.orkg import PAPER_COLUMNS, _paper_fields


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _tables(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tables = {}
    for table in payload["tables"]:
        key = table["table_key"]
        if not isinstance(key, str) or not key or key in tables:
            raise ValueError(f"Invalid or duplicate Step 7 table key: {key!r}.")
        tables[key] = table
    return tables


def _aligned(table: dict[str, Any], alignment: dict[str, Any]) -> dict[str, Any]:
    """Apply a caller-reviewed evaluation crosswalk, never an inference repair."""
    result = deepcopy(table)
    spec = alignment.get("tables", {}).get(table["table_key"], {})
    result["table_key"] = spec.get("table_key", table["table_key"])

    def translate(value: Any, mapping: dict[str, Any] | None) -> Any:
        if mapping is None or value is None:
            return value
        return mapping.get(str(value), ("unmapped", value))

    row_map = spec.get("rows")
    column_map = spec.get("columns")
    bibliography_map = alignment.get("bibliography_indices")

    def coordinate(pair: list[Any]) -> list[Any]:
        return [translate(pair[0], row_map), translate(pair[1], column_map)]

    for column in result.get("columns", []):
        column["source_columns"] = [translate(c, column_map) for c in column["source_columns"]]
        column["header_cells"] = [coordinate(cell) for cell in column["header_cells"]]
    for record in result.get("records", []):
        record["source_rows"] = [translate(r, row_map) for r in record["source_rows"]]
        for cell in record["cells"]:
            cell["sources"] = [coordinate(source) for source in cell["sources"]]
            for rule in cell["propagations"]:
                rule["target_row"] = translate(rule["target_row"], row_map)
                rule["source"] = coordinate(rule["source"])
            for blank in cell["blanks"]:
                blank["row"] = translate(blank["row"], row_map)
        for link in record["links"]:
            link["source_row"] = translate(link["source_row"], row_map)
            link["target_row"] = translate(link["target_row"], row_map)
            link["source_column"] = translate(link["source_column"], column_map)
            link["reference_index"] = translate(link["reference_index"], bibliography_map)
    return result


def _label(column: dict[str, Any]) -> str:
    return " ".join(column["label"].casefold().split())


def _link_key(link: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(link[key] for key in (
        "target_row", "source_row", "source_column", "reference_index", "occurrence", "kind",
    ))


def _projection_intact(directory: Path, table: dict[str, Any]) -> bool:
    path = (directory / table["csv"]).resolve()
    if path.parent != directory.resolve():
        return False
    try:
        rows = read_csv(path)
    except (OSError, UnicodeError, csv.Error):
        return False
    headers = [f"{c['label']}<{c['datatype']}>" for c in table["columns"]] + list(PAPER_COLUMNS)
    if not rows or rows[0] != headers or len(rows) - 1 != table["export_rows"]:
        return False
    seen = set()
    for record in table["records"]:
        scientific = [cell["value"] for cell in record["cells"]]
        for link in record["links"]:
            if "output_row" not in link:
                continue
            row = link["output_row"]
            fields = _paper_fields(link.get("identity"))
            if fields is None or type(row) is not int or not 1 <= row < len(rows) or row in seen:
                return False
            if scientific + fields != rows[row]:
                return False
            seen.add(row)
    return len(seen) == len(rows) - 1


def _score(counter: Counter[str], expected: Iterable[Any], observed: Iterable[Any]) -> None:
    gold, pred = Counter(expected), Counter(observed)
    tp = sum((gold & pred).values())
    counter.update(tp=tp, fn=sum(gold.values()) - tp, fp=sum(pred.values()) - tp)


def _prf(counter: Counter[str]) -> dict[str, Any]:
    tp, fp, fn = (counter[k] for k in ("tp", "fp", "fn"))
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
    }


def _field_ok(field: str, gold: Any, pred: Any) -> bool:
    if pred is None or pred == "" or pred == []:
        return False
    if field == "canonical_doi":
        return normalize_doi(str(gold)) == normalize_doi(str(pred))
    if field == "canonical_title":
        return _title_ok(gold, pred)
    if field == "canonical_authors":
        return bool(_authors(gold)) and _authors(gold) == _authors(pred)
    if field == "canonical_venue":
        return _venue_ok(gold, pred)
    return str(gold).strip() == str(pred).strip()


def _benchmark_gold(paper: Path) -> dict[str, Any]:
    """Read independent evidence, with draft structure labeled as development data."""
    step5 = _read(paper / "step5_reference_matching" / "gold.json")
    step6 = _read(paper / "step6_reference_resolution" / "gold.json")
    bibliography_file = paper / "step4_bibliography" / "gold.json"
    bibliography = {int(item["nr"]): item["ref"] for item in json.loads(
        bibliography_file.read_text(encoding="utf-8-sig")
    )}
    identities = {}
    for record in step6["records"]:
        index = record.get("gold_index")
        if (type(index) is not int or index <= 0 or index in identities
                or index not in bibliography or record.get("raw_reference") != bibliography[index]):
            raise ValueError(f"Invalid or unverified Step 6 gold_index: {index!r}.")
        identities[index] = {
            f"canonical_{field}": record.get(f"gold_{field}")
            for field in ("doi", "title", "authors", "year", "venue")
        }

    directory = paper / "step7_orkg_gold"
    projection = _read(directory / "manifest.json")
    draft = {table["table_key"]: table for table in projection["tables"]}
    provenance_path = directory / "provenance.csv"
    by_table = defaultdict(list)
    if provenance_path.is_file():
        with provenance_path.open(newline="", encoding="utf-8-sig") as handle:
            for entry in csv.DictReader(handle):
                by_table[entry["table_key"]].append(entry)

    tables = []
    for table in step5["tables"]:
        key = table["table_key"]
        info = draft[key]
        path = directory / (info.get("output_orkg_csv") or info["output_csv"])
        values = read_csv(path)
        positions = [i for i, name in enumerate(values[0]) if not name.startswith("paper:")]
        columns = [
            {"label": re.sub(r"<[^>]+>$", "", values[0][i])} for i in positions
        ]
        links = [
            {
                "source_row": cell["row_index"], "target_row": cell["row_index"],
                "source_column": cell["column_index"], "reference_index": index,
                "occurrence": occurrence, "kind": "explicit", "identity": identities.get(index, {}),
            }
            for cell in table["citation_cells"]
            for occurrence, index in enumerate(cell["bibliography_indices"])
        ]
        expected = {
            "table_key": key, "columns": columns, "links": links,
            "inheritance_annotated": key in by_table,
        }
        if key in by_table:
            records: dict[int, dict[str, Any]] = {}
            for item in by_table[key]:
                row = int(item["source_csv_row"]) - 1
                output_row = int(item["output_csv_row"]) - 1
                index = int(item["gold_bibliography_index"])
                inherited = item["association_evidence"] == "expert_confirmed_inherited_link"
                if inherited:
                    anchor = int(item["inherited_from_source_csv_row"]) - 1
                    anchor_links = [
                        link for link in links if link["kind"] == "explicit"
                        and link["source_row"] == anchor and link["reference_index"] == index
                    ]
                    if len(anchor_links) != 1:
                        raise ValueError(f"Inheritance lacks a unique gold source in {key}, row {row}.")
                    links.append({**anchor_links[0], "kind": "inherited", "target_row": row})
                cells = [{"value": values[output_row][i]} for i in positions]
                entry = records.setdefault(row, {"source_rows": [row], "cells": cells, "expansion": 0})
                if entry["cells"] != cells:
                    raise ValueError(f"Conflicting scientific values in {key}, row {row}.")
                entry["expansion"] += 1
            expected["records"] = list(records.values())
        tables.append(expected)
    return {
        "schema_version": 1,
        "split": "development",
        "index_namespace": "gold_index",
        "notes": (
            "Draft ORKG headers and provenance-backed rows are development checks. "
            "Routing, hierarchical column lineage and propagation need independent annotation. "
            "Identity expectations come from validated Step 6 gold_index."
        ),
        "tables": tables,
    }


def evaluate_step7(
    prediction_path: Path,
    *,
    gold_path: Path | None = None,
    benchmark_paper: Path | None = None,
    alignment_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate annotated properties only, leaving unscored denominators explicit."""
    if (gold_path is None) == (benchmark_paper is None):
        raise ValueError("Supply one evaluation gold file or benchmark paper directory.")
    prediction = _read(prediction_path)
    gold = _read(gold_path) if gold_path is not None else _benchmark_gold(benchmark_paper)
    if gold.get("schema_version") != 1 or prediction.get("schema_version") != 1:
        raise ValueError("Unsupported Step 7 evaluation or prediction schema.")
    if gold.get("split") not in ("development", "held_out"):
        raise ValueError("Evaluation gold must declare development or held_out.")

    expected = _tables(gold)
    alignment = _read(alignment_path) if alignment_path is not None else {}
    gold_namespace = gold.get("index_namespace")
    if (prediction["tables"] and gold_namespace
            and prediction.get("index_namespace") != gold_namespace
            and "bibliography_indices" not in alignment):
        raise ValueError("Different bibliography namespaces require an evaluation-only crosswalk.")
    observed = _tables({
        "tables": [_aligned(t, alignment) for t in _tables(prediction).values()]
        if alignment else list(_tables(prediction).values())
    })
    scores: dict[str, Counter[str]] = defaultdict(Counter)
    routing: dict[str, Counter[str]] = defaultdict(Counter)
    exact: dict[str, Counter[str]] = defaultdict(Counter)
    identity: dict[str, Counter[str]] = defaultdict(Counter)
    coverage: Counter[str] = Counter()
    omitted: Counter[str] = Counter()
    intact = set()
    for pred in observed.values():
        coverage[pred.get("status", "missing_status")] += 1
        coverage["projected_rows"] += pred.get("export_rows", 0)
        omitted.update(item["reason"] for item in pred.get("omissions", []))
        if "csv" in pred:
            if _projection_intact(prediction_path.parent, pred):
                intact.add(pred["table_key"])
            else:
                coverage["missing_or_inconsistent_csv"] += 1
    for key, truth in expected.items():
        pred = observed.get(key, {})
        if "route" in truth:
            route = truth["route"]
            if route not in ("deterministic", "llm_assisted"):
                raise ValueError(f"Invalid gold route for {key}.")
            routing[route][pred.get("route", "missing")] += 1

        gold_columns, pred_columns = truth.get("columns", []), pred.get("columns", [])
        if "columns" in truth:
            gold_labels, pred_labels = ([_label(c) for c in columns]
                                        for columns in (gold_columns, pred_columns))
            _score(scores["column_correspondence"], gold_labels, pred_labels)
            exact["column_order"]["total"] += 1
            exact["column_order"]["correct"] += gold_labels == pred_labels
            predicted_columns = {_label(c): c for c in pred_columns}
            for column in gold_columns:
                observed_column = predicted_columns.get(_label(column), {})
                for field, metric in (("source_columns", "column_lineage"),
                                      ("header_cells", "header_hierarchy"), ("datatype", "datatype")):
                    if field in column:
                        exact[metric]["total"] += 1
                        exact[metric]["correct"] += column[field] == observed_column.get(field)

        gold_records, pred_records = truth.get("records", []), pred.get("records", [])
        if "records" in truth:
            gold_rows, pred_rows = ([tuple(r["source_rows"]) for r in records]
                                    for records in (gold_records, pred_records))
            _score(scores["row_correspondence"], gold_rows, pred_rows)
            exact["row_order"]["total"] += 1
            exact["row_order"]["correct"] += gold_rows == pred_rows
            predicted_records = {tuple(r["source_rows"]): r for r in pred_records}
            for record in gold_records:
                found = predicted_records.get(tuple(record["source_rows"]), {})
                predicted_cells = {_label(c): cell for c, cell in zip(pred_columns, found.get("cells", []))}
                for column, cell in zip(gold_columns, record.get("cells", [])):
                    observed_cell = predicted_cells.get(_label(column), {})
                    for field, metric in (("value", "cell_values"), ("sources", "cell_placement"),
                                          ("blanks", "blank_disposition")):
                        if field in cell:
                            exact[metric]["total"] += 1
                            exact[metric]["correct"] += cell[field] == observed_cell.get(field)
                    if "propagations" in cell:
                        def rule_key(rule: dict[str, Any]) -> tuple[Any, ...]:
                            return (rule["target_row"], tuple(rule["source"]), rule["mode"],
                                    tuple(rule.get("slice") or []))
                        _score(scores["value_propagation"], map(rule_key, cell["propagations"]),
                               map(rule_key, observed_cell.get("propagations", [])))
                if "expansion" in record:
                    exact["row_expansion"]["total"] += 1
                    exact["row_expansion"]["correct"] += (
                        bool(found) and len(found.get("links", [])) == record["expansion"]
                    )
                if "projection_rows" in record:
                    exact["projection_row_count"]["total"] += 1
                    exact["projection_row_count"]["correct"] += bool(found) and sum(
                        "output_row" in link for link in found.get("links", [])
                    ) == record["projection_rows"]

        if "links" in truth or any("links" in record for record in gold_records):
            gold_links = truth.get("links", [link for r in gold_records for link in r.get("links", [])])
            pred_links = [link for record in pred_records for link in record.get("links", [])]
            assess_inheritance = truth.get("inheritance_annotated", True)
            for kind in ("explicit", "inherited"):
                if kind == "inherited" and not assess_inheritance:
                    continue
                _score(scores[f"citations_{kind}"],
                       (_link_key(link) for link in gold_links if link["kind"] == kind),
                       (_link_key(link) for link in pred_links if link["kind"] == kind))
            resolved = {_link_key(link): link.get("identity", {}) for link in pred_links}
            for link in gold_links:
                aligned_identity = resolved.get(_link_key(link))
                for field, value in link.get("identity", {}).items():
                    if value is None or value == "" or value == []:
                        continue
                    stats = identity[field]
                    stats["gold_covered"] += 1
                    if aligned_identity is not None:
                        stats["citation_aligned"] += 1
                        actual = aligned_identity.get(field)
                        stats["value_present"] += actual is not None and actual != "" and actual != []
                        stats["correct"] += _field_ok(field, value, actual)

    comparisons = {
        name: {"correct": count["correct"], "total": count["total"],
               "accuracy": count["correct"] / count["total"] if count["total"] else None}
        for name, count in exact.items()
    }
    invocation_count = sum(t.get("llm_invocations", 0) for t in observed.values())
    usage = [t.get("llm", {}).get("usage", {}) for t in observed.values() if "llm" in t]
    tokens = {
        field: sum(item[field] for item in usage if type(item.get(field)) is int)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    route_total = sum(sum(count.values()) for count in routing.values())
    route_correct = sum(count[label] for label, count in routing.items())
    return {
        "schema_version": 1, "split": gold["split"], "notes": gold.get("notes"),
        "routing": {"correct": route_correct, "total": route_total,
                    "accuracy": route_correct / route_total if route_total else None,
                    "confusion": routing},
        "structure_and_citations": {name: _prf(count) for name, count in scores.items()},
        "structural_comparisons": comparisons,
        "identity": {
            name: {"gold_covered": count["gold_covered"], "citation_aligned": count["citation_aligned"],
                   "value_present": count["value_present"],
                   "correct": count["correct"],
                   "conditional_accuracy": count["correct"] / count["citation_aligned"]
                   if count["citation_aligned"] else None,
                   "end_to_end_recall": count["correct"] / count["gold_covered"]}
            for name, count in identity.items()
        },
        "coverage": {
            "gold_fragments": len(expected), "predicted_fragments": len(observed),
            "missing_fragments": sorted(expected.keys() - observed.keys()),
            "extra_fragments": sorted(observed.keys() - expected.keys()),
            "gold_fragments_exported": sum(
                observed[key].get("status") == "exported" for key in expected.keys() & intact
            ),
            "counts": coverage, "omissions": omitted,
        },
        "runtime_seconds": prediction.get("runtime_seconds"),
        "llm": {"invocations": invocation_count, "responses": len(usage), "reported_tokens": tokens},
    }
