
"""Evidence, routing and deterministic execution for Step 7 structure plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tabulus.reference_matching.matching import normalize_text
from tabulus.table_ocr.parsing import (
    _expand_html_spans,
    _SimpleTableHTMLParser,
    extract_html_tables,
    extract_markdown_text,
    otsl_table_to_html,
)


@dataclass(frozen=True)
class StructureEvidence:
    """Native geometry is usable only when it aligns to the exported grid."""

    reasons: tuple[str, ...] = ()
    spans: tuple[dict[str, Any], ...] = ()


def native_structure(rows: list[list[str]], native: dict[str, Any]) -> StructureEvidence:
    reasons = set()
    spans = []
    normalized_rows = [[normalize_text(value) for value in row] for row in rows]
    width = max(map(len, rows), default=0)
    for value in native.get("native_markdown") or []:
        markup = extract_markdown_text(value)
        if "<fcel>" in markup or "<nl>" in markup:
            markup = otsl_table_to_html(markup)
        for html in extract_html_tables(markup):
            parser = _SimpleTableHTMLParser()
            parser.feed(html)
            cells = parser.rows
            has_span = any(cell.rowspan > 1 or cell.colspan > 1 for row in cells for cell in row)
            if has_span:
                reasons.add("native_merged_cells")
            header = re.search(r"<thead\b[^>]*>(.*?)</thead>", html, re.I | re.S)
            if header and len(re.findall(r"<tr\b", header[1], re.I)) > 1:
                reasons.add("native_hierarchical_header")
            if not has_span:
                continue
            if len(cells) != len(rows) or any(
                sum(cell.colspan for cell in row) > width for row in cells
            ):
                reasons.add("native_grid_unaligned")
                continue
            grid = _expand_html_spans(cells)
            if [[normalize_text(value) for value in row] for row in grid] != normalized_rows:
                reasons.add("native_grid_unaligned")
                continue

            occupied: set[tuple[int, int]] = set()
            for row, cell_row in enumerate(cells):
                column = 0
                for cell in cell_row:
                    end = min(row + cell.rowspan, len(cells))
                    while any(
                        (r, c) in occupied
                        for r in range(row, end)
                        for c in range(column, column + cell.colspan)
                    ):
                        column += 1
                    if cell.rowspan > 1 or cell.colspan > 1:
                        spans.append(
                            {"source": [row, column], "row_end": end,
                             "column_end": column + cell.colspan}
                        )
                    occupied.update(
                        (r, c)
                        for r in range(row, end)
                        for c in range(column, column + cell.colspan)
                    )
                    column += cell.colspan
    return StructureEvidence(tuple(sorted(reasons)), tuple(spans))


def routing_reasons(table: dict[str, Any], evidence: StructureEvidence) -> tuple[str, ...]:
    rows = table["rows"]
    reasons = list(evidence.reasons)
    if not rows or table["header_row_index"] != 0:
        reasons.append("unverified_single_header")
    if not rows:
        return tuple(reasons)
    if any(len(row) != len(rows[0]) for row in rows):
        reasons.append("ragged_source_grid")

    ref = table["reference_column_index"]
    if type(ref) is not int or not 0 <= ref < len(rows[0]):
        reasons.append("unverified_reference_column")
    headings = [normalize_text(value).casefold() for i, value in enumerate(rows[0]) if i != ref]
    if not headings or "" in headings or len(set(headings)) != len(headings):
        reasons.append("missing_or_duplicate_scientific_headers")
    if any(not value.strip() for row in rows[1:] for i, value in enumerate(row) if i != ref):
        reasons.append("ambiguous_scientific_blanks")
    if any(not row[ref].strip() for row in rows[1:] if type(ref) is int and 0 <= ref < len(row)):
        reasons.append("blank_reference_cells")
    if any([normalize_text(value).casefold() for value in row] ==
           [normalize_text(value).casefold() for value in rows[0]] for row in rows[1:]):
        reasons.append("repeated_header")
    return tuple(reasons)


def direct_plan(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "header_rows": [0],
        "columns": [
            {"label": value, "header_cells": [[0, column]], "source_columns": [column]}
            for column, value in enumerate(table["rows"][0])
            if column != table["reference_column_index"]
        ],
    }


def _object(value: Any, required: set[str], optional: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError(f"Invalid fields in {label}.")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array.")
    return value


def _index(value: Any, limit: int, label: str) -> int:
    if type(value) is not int or not 0 <= value < limit:
        raise ValueError(f"{label} is outside the source bounds.")
    return value


def _interval(value: Any, limit: int, label: str) -> range:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(x) is not int for x in value) or not 0 <= value[0] < value[1] <= limit):
        raise ValueError(f"Invalid half-open {label} range.")
    return range(*value)


def _cell(rows: list[list[str]], coordinate: Any) -> tuple[int, int]:
    if not isinstance(coordinate, list) or len(coordinate) != 2:
        raise ValueError("A source cell must be [row, column].")
    row = _index(coordinate[0], len(rows), "Source row")
    return row, _index(coordinate[1], len(rows[row]), "Source column")


def _range_with_anchor(rule: dict[str, Any], anchor: int, count: int) -> tuple[range, range]:
    targets = _interval(rule["rows"], count, "target row")
    group = _interval(rule["group"], count, "group")
    if not group.start <= anchor < targets.start or targets.stop > group.stop:
        raise ValueError("The anchor must precede its targets inside the declared group.")
    return targets, group


def execute_plan(
    table: dict[str, Any], proposal: Any, evidence: StructureEvidence = StructureEvidence(),
) -> dict[str, Any]:
    """Validate coordinates and derive all exported scientific values from source cells."""
    plan = _object(
        proposal, {"schema_version", "header_rows", "columns"},
        {"group_rows", "groups", "merge_rows", "propagations", "inherit_references", "blanks"},
        "structure plan",
    )
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("Unsupported structure plan schema version.")

    rows = table["rows"]
    count = len(rows)
    width = max(map(len, rows), default=0)
    ref = table["reference_column_index"]
    matches = {
        row: match["matched_reference_indices"]
        for row, match in table["matches_by_row"].items()
        if row != table["header_row_index"] and match.get("matched_reference_indices")
    }

    header_rows = [_index(r, count, "Header row") for r in _array(plan["header_rows"], "header_rows")]
    group_rows = [_index(r, count, "Group row") for r in _array(plan.get("group_rows", []), "group_rows")]
    structural_rows = set(header_rows) | set(group_rows)
    if len(structural_rows) != len(header_rows) + len(group_rows) or structural_rows & matches.keys():
        raise ValueError("Structural rows must be distinct and may not discard explicit links.")
    recorded_header = table["header_row_index"]
    if recorded_header is not None and recorded_header not in header_rows:
        raise ValueError("The recorded Step 5 header must remain a header.")
    if type(ref) is int and any(
        ref < len(rows[r]) and rows[r][ref].strip() for r in group_rows
    ):
        raise ValueError("A group heading cannot remove a nonblank reference cell.")
    if any(
        r != recorded_header and match.get("unmatched_tokens")
        for r, match in table["matches_by_row"].items() if r in structural_rows
    ):
        raise ValueError("Structural rows cannot discard unmatched citation evidence.")

    # Declared heading groups may nest, such as category > method > reactant.
    # Intersecting scopes with neither containing the other are ambiguous.
    group_scopes = {}
    for item in _array(plan.get("groups", []), "groups"):
        scope = _interval(item, count, "heading group")
        if scope.start not in group_rows or scope.start in group_scopes or len(scope) < 2:
            raise ValueError("Each heading group must start at a distinct group row and include data.")
        if any(
            scope.start < other.start < scope.stop < other.stop
            or other.start < scope.start < other.stop < scope.stop
            for other in group_scopes.values()
        ):
            raise ValueError("Heading groups must be disjoint or nested.")
        group_scopes[scope.start] = scope

    columns = []
    labels = set()
    covered_columns = set()
    for item in _array(plan["columns"], "columns"):
        spec = _object(item, {"label", "source_columns", "header_cells"}, {"datatype"}, "column")
        label = spec["label"]
        if (not isinstance(label, str) or not label.strip() or "<" in label or ">" in label
                or label.strip().casefold().startswith(("paper:", "contribution:"))
                or label.strip().casefold() in labels):
            raise ValueError("Scientific column labels must be unique, nonempty and untyped.")
        labels.add(label.strip().casefold())
        sources = [_index(c, width, "Scientific column") for c in _array(spec["source_columns"], "source_columns")]
        if ref in sources or len(sources) != len(set(sources)):
            raise ValueError("Source columns must be distinct scientific columns.")
        covered_columns.update(sources)
        headers = [_cell(rows, c) for c in _array(spec["header_cells"], "header_cells")]
        for r, c in headers:
            if r not in header_rows or c == ref or not rows[r][c].strip():
                raise ValueError(
                    f"Column {label!r} cites invalid header evidence at [{r}, {c}]. "
                    "Use a nonblank scientific header cell or leave header_cells empty."
                )
        datatype = spec.get("datatype", "Text")
        if datatype not in ("Text", "Integer", "Decimal", "Boolean"):
            raise ValueError("Unsupported ORKG scientific datatype.")
        columns.append({
            "label": label, "datatype": datatype, "source_columns": sources,
            "header_cells": [list(c) for c in headers],
        })
    if not columns:
        raise ValueError("A structure plan must include scientific columns.")
    if any(
        value.strip() and c != ref and c not in covered_columns
        for r, row in enumerate(rows) if r not in structural_rows
        for c, value in enumerate(row)
    ):
        raise ValueError("A nonblank scientific data column is missing from the plan.")

    # The default record is one physical data row. Multirow records are opt-in.
    merged = {}
    merged_rows = set()
    for item in _array(plan.get("merge_rows", []), "merge_rows"):
        group = _interval(item, count, "record")
        if len(group) < 2 or (merged_rows | structural_rows).intersection(group):
            raise ValueError("Merged records must contain distinct contiguous data rows.")
        merged[group.start] = list(group)
        merged_rows.update(group)

    fills = {}
    for item in _array(plan.get("propagations", []), "propagations"):
        rule = _object(item, {"source", "column", "rows", "group"}, {"mode", "slice"}, "propagation")
        row, column = _cell(rows, rule["source"])
        target_column = _index(rule["column"], len(columns), "Target column")
        targets, group = _range_with_anchor(rule, row, count)
        declared = group_scopes.get(row)
        if declared is not None and (group.start < declared.start or group.stop > declared.stop):
            raise ValueError("A propagation exceeds its heading's group scope.")
        intervening = structural_rows.intersection(range(row + 1, group.stop))
        if column == ref or row in header_rows or any(
            heading not in group_scopes or group_scopes[heading].stop > group.stop
            for heading in intervening
        ):
            raise ValueError("A propagation cannot cross an unrelated heading or copy a reference cell.")
        text = rows[row][column]
        if "slice" in rule:
            part = _interval(rule["slice"], len(text), "source text")
            text = text[part.start:part.stop]
        mode = rule.get("mode", "fill")
        if not text.strip() or mode not in ("fill", "prefix"):
            raise ValueError("A propagation requires nonblank source evidence and a known mode.")

        source_cols = columns[target_column]["source_columns"]
        # The anchor establishes the current group. An intervening occupied
        # destination cell invalidates a fill-down, even before its first target.
        if mode == "fill" and any(
            c < len(rows[r]) and rows[r][c].strip()
            for r in range(row + 1, targets.stop) if r not in structural_rows for c in source_cols
        ):
            raise ValueError("A fill-down crosses a nonblank destination cell.")
        for target in targets:
            if target in structural_rows:
                continue
            key = target, target_column
            if key in fills:
                raise ValueError("Overlapping propagations for one target cell.")
            fills[key] = (text, [row, column], mode, rule.get("slice"))

    blanks = {}
    for item in _array(plan.get("blanks", []), "blanks"):
        rule = _object(item, {"column", "rows", "kind"}, set(), "blank disposition")
        column = _index(rule["column"], len(columns), "Blank target column")
        targets = _interval(rule["rows"], count, "blank")
        if rule["kind"] not in ("intentional", "unknown") or structural_rows.intersection(targets):
            raise ValueError("A blank disposition must describe data rows and a known kind.")
        for row in targets:
            key = row, column
            if key in blanks or key in fills or any(
                c < len(rows[row]) and rows[row][c].strip() for c in columns[column]["source_columns"]
            ):
                raise ValueError("Blank dispositions require distinct blank cells.")
            blanks[key] = rule["kind"]

    inherited = {}
    for item in _array(plan.get("inherit_references", []), "inherit_references"):
        rule = _object(
            item, {"source_row", "rows", "group"}, {"occurrences", "evidence_cells"},
            "citation inheritance",
        )
        anchor = _index(rule["source_row"], count, "Citation source row")
        targets, group = _range_with_anchor(rule, anchor, count)
        anchor_links = matches.get(anchor)
        if not anchor_links or type(ref) is not int or structural_rows.intersection(group):
            raise ValueError("An inheritance group must have an explicit anchor and only data rows.")
        if any(r in matches or (ref < len(rows[r]) and rows[r][ref].strip())
               for r in range(anchor + 1, targets.stop)):
            raise ValueError("Citation inheritance crosses an intervening citation.")
        group_evidence = [
            _cell(rows, cell) for cell in _array(rule.get("evidence_cells", []), "evidence_cells")
        ]
        reference_span = any(
            span["source"] == [anchor, ref] and span["row_end"] >= targets.stop
            for span in evidence.spans
        )
        scientific_anchor = any(
            row in group and row <= anchor and column != ref and rows[row][column].strip()
            and all(column < len(rows[r]) and not rows[r][column].strip()
                    for r in range(row + 1, targets.stop))
            for row, column in group_evidence
        )
        if not reference_span and not scientific_anchor:
            raise ValueError("Citation inheritance needs an aligned reference span or scientific group evidence.")
        occurrences = _array(rule.get("occurrences", list(range(len(anchor_links)))), "occurrences")
        occurrences = [_index(i, len(anchor_links), "Citation occurrence") for i in occurrences]
        if not occurrences or occurrences != sorted(set(occurrences)):
            raise ValueError("Citation occurrences must be unique and in source order.")
        for target in targets:
            if target in inherited:
                raise ValueError("Overlapping citation inheritance.")
            inherited[target] = (anchor, occurrences)

    records = []
    for row in range(count):
        if row in structural_rows or (row in merged_rows and row not in merged):
            continue
        record_rows = merged.get(row, [row])
        cells = []
        links = []
        for column, spec in enumerate(columns):
            parts, origins, operations, dispositions = [], [], [], []
            for r in record_rows:
                values = [
                    (rows[r][c], [r, c]) for c in spec["source_columns"] if c < len(rows[r])
                ]
                copied = fills.get((r, column))
                if copied is not None:
                    text, anchor, mode, selection = copied
                    operation = {"target_row": r, "source": anchor, "mode": mode, "slice": selection}
                    if mode == "prefix":
                        populated = [(value, coordinate) for value, coordinate in values if value.strip()]
                        if populated:
                            parts.append(f"{text}: " + " | ".join(value for value, _ in populated))
                            origins.extend([anchor, *(coordinate for _, coordinate in populated)])
                            operations.append(operation)
                            continue
                    else:
                        values = [(text, anchor)]
                        operations.append(operation)
                for text, coordinate in values:
                    if text.strip() or len(record_rows) == len(values) == 1:
                        parts.append(text)
                        origins.append(coordinate)
                if not any(text.strip() for text, _ in values):
                    dispositions.append({"row": r, "kind": blanks.get((r, column), "unknown")})
            value = " | ".join(parts)
            datatype = spec["datatype"]
            if value.strip() and (
                (datatype == "Integer" and not re.fullmatch(r"[+-]?[0-9]+", value.strip()))
                or (datatype == "Decimal" and not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", value.strip()))
                or (datatype == "Boolean" and value.strip().lower() not in ("true", "false"))
            ):
                raise ValueError(f"Row {row} has a value inconsistent with {datatype}.")
            cells.append({"value": value, "sources": origins, "propagations": operations, "blanks": dispositions})

        for r in record_rows:
            anchor, occurrences = inherited.get(r, (r, list(range(len(matches.get(r, []))))))
            for occurrence in occurrences:
                links.append({
                    "source_row": anchor, "target_row": r, "source_column": ref,
                    "occurrence": occurrence, "reference_index": matches[anchor][occurrence],
                    "kind": "explicit" if anchor == r else "inherited",
                })
        records.append({"source_rows": record_rows, "cells": cells, "links": links})
    return {"columns": columns, "records": records}
