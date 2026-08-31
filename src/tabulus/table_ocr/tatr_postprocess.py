"""Deterministic Table Transformer structure post-processing.

Adapted from Microsoft Table Transformer ``src/postprocess.py``.

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The implementation is intentionally local so Tabulus can combine TATR structure
predictions with externally supplied OCR tokens without cloning the upstream
repository at runtime.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pymupdf


Rect = pymupdf.Rect


def _area(bbox: list[float]) -> float:
    return Rect(bbox).get_area()


def iob(bbox1: list[float], bbox2: list[float]) -> float:
    """Intersection area over the area of ``bbox1``."""
    rect1 = Rect(bbox1)
    area1 = rect1.get_area()
    if area1 <= 0:
        return 0.0
    return rect1.intersect(bbox2).get_area() / area1


def sort_objects_by_score(objects: list[dict[str, Any]], reverse: bool = True) -> list[dict[str, Any]]:
    sign = -1 if reverse else 1
    return sorted(objects, key=lambda item: sign * float(item["score"]))


def sort_objects_left_to_right(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(objects, key=lambda item: item["bbox"][0] + item["bbox"][2])


def sort_objects_top_to_bottom(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(objects, key=lambda item: item["bbox"][1] + item["bbox"][3])


def slot_into_containers(
    containers: list[dict[str, Any]],
    packages: list[dict[str, Any]],
    *,
    overlap_threshold: float = 0.5,
    unique_assignment: bool = True,
    forced_assignment: bool = False,
) -> tuple[list[list[int]], list[list[int]], list[float]]:
    container_assignments = [[] for _ in containers]
    package_assignments = [[] for _ in packages]
    best_scores: list[float] = []

    if not containers or not packages:
        return container_assignments, package_assignments, best_scores

    for package_num, package in enumerate(packages):
        package_rect = Rect(package["bbox"])
        package_area = package_rect.get_area()
        if package_area <= 0:
            best_scores.append(0.0)
            continue

        scores: list[tuple[float, int]] = []
        for container_num, container in enumerate(containers):
            overlap = Rect(container["bbox"]).intersect(package["bbox"]).get_area() / package_area
            scores.append((overlap, container_num))
        scores.sort(reverse=True)

        best_score, best_container = scores[0]
        best_scores.append(best_score)
        if forced_assignment or best_score >= overlap_threshold:
            container_assignments[best_container].append(package_num)
            package_assignments[package_num].append(best_container)

        if not unique_assignment:
            for score, container_num in scores[1:]:
                if score < overlap_threshold:
                    break
                container_assignments[container_num].append(package_num)
                package_assignments[package_num].append(container_num)

    return container_assignments, package_assignments, best_scores


def extract_text_from_spans(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return ""

    ordered = list(spans)
    ordered.sort(key=lambda span: int(span.get("span_num", 0)))
    ordered.sort(key=lambda span: int(span.get("line_num", 0)))
    ordered.sort(key=lambda span: int(span.get("block_num", 0)))

    line_texts: list[str] = []
    current: list[str] = [str(ordered[0].get("text", ""))]

    for first, second in zip(ordered[:-1], ordered[1:]):
        same_line = (
            first.get("block_num") == second.get("block_num")
            and first.get("line_num") == second.get("line_num")
        )
        if same_line:
            current.append(str(second.get("text", "")))
        else:
            line_texts.append(" ".join(current).strip())
            current = [str(second.get("text", ""))]

    line_texts.append(" ".join(current).strip())
    return " ".join(text for text in line_texts if text).strip()


def _span_subset(spans: list[dict[str, Any]], bbox: list[float], threshold: float = 0.5) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    target = Rect(bbox)
    for span in spans:
        span_rect = Rect(span["bbox"])
        span_area = span_rect.get_area()
        if span_area <= 0:
            continue
        if span_rect.intersect(target).get_area() / span_area >= threshold:
            selected.append(span)
    return selected


def _remove_objects_without_content(spans: list[dict[str, Any]], objects: list[dict[str, Any]]) -> None:
    for obj in objects[:]:
        if not extract_text_from_spans(_span_subset(spans, obj["bbox"])):
            objects.remove(obj)


def _nms_by_containment(
    containers: list[dict[str, Any]],
    packages: list[dict[str, Any]],
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    containers = sort_objects_by_score(containers)
    assignments, _, _ = slot_into_containers(
        containers,
        packages,
        overlap_threshold=overlap_threshold,
        unique_assignment=True,
        forced_assignment=False,
    )
    suppressed = [False] * len(containers)

    for second in range(1, len(containers)):
        second_packages = set(assignments[second])
        if not second_packages:
            suppressed[second] = True
        for first in range(second):
            if suppressed[first]:
                continue
            if second_packages.intersection(assignments[first]):
                suppressed[second] = True
                break

    return [obj for idx, obj in enumerate(containers) if not suppressed[idx]]


def _nms(
    objects: list[dict[str, Any]],
    *,
    match_criteria: str = "object2_overlap",
    match_threshold: float = 0.05,
    keep_higher: bool = True,
) -> list[dict[str, Any]]:
    if not objects:
        return []

    objects = sort_objects_by_score(objects, reverse=keep_higher)
    suppressed = [False] * len(objects)

    for second in range(1, len(objects)):
        rect2 = Rect(objects[second]["bbox"])
        area2 = rect2.get_area()
        for first in range(second):
            if suppressed[first]:
                continue
            rect1 = Rect(objects[first]["bbox"])
            area1 = rect1.get_area()
            intersection = rect1.intersect(rect2).get_area()
            if match_criteria == "object1_overlap":
                metric = intersection / area1 if area1 > 0 else 0.0
            elif match_criteria == "iou":
                denom = area1 + area2 - intersection
                metric = intersection / denom if denom > 0 else 0.0
            else:
                metric = intersection / area2 if area2 > 0 else 0.0
            if metric >= match_threshold:
                suppressed[second] = True
                break

    return [obj for idx, obj in enumerate(objects) if not suppressed[idx]]


def _refine_rows(rows: list[dict[str, Any]], tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if tokens:
        rows = _nms_by_containment(rows, tokens, 0.5)
        _remove_objects_without_content(tokens, rows)
    else:
        rows = _nms(rows, match_criteria="object2_overlap", match_threshold=0.5)
    return sort_objects_top_to_bottom(rows) if len(rows) > 1 else rows


def _refine_columns(columns: list[dict[str, Any]], tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if tokens:
        columns = _nms_by_containment(columns, tokens, 0.5)
        _remove_objects_without_content(tokens, columns)
    else:
        columns = _nms(columns, match_criteria="object2_overlap", match_threshold=0.25)
    return sort_objects_left_to_right(columns) if len(columns) > 1 else columns


def _align_headers(headers: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row["header"] = False

    header_rows: list[int] = []
    for header in headers:
        for row_num, row in enumerate(rows):
            row_height = row["bbox"][3] - row["bbox"][1]
            if row_height <= 0:
                continue
            overlap = min(row["bbox"][3], header["bbox"][3]) - max(row["bbox"][1], header["bbox"][1])
            if overlap / row_height >= 0.5:
                header_rows.append(row_num)

    if not header_rows:
        return []

    header_rows = sorted(set(header_rows))
    if header_rows[0] > 0:
        header_rows = list(range(header_rows[0] + 1)) + header_rows

    header_rect = Rect()
    last = -1
    for row_num in header_rows:
        if row_num != last + 1:
            break
        rows[row_num]["header"] = True
        header_rect.include_rect(rows[row_num]["bbox"])
        last = row_num

    return [{"bbox": list(header_rect)}]


def _align_supercells(
    supercells: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    columns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []

    for supercell in supercells:
        supercell["header"] = False
        header_rows: set[int] = set()
        data_rows: set[int] = set()

        for row_num, row in enumerate(rows):
            row_height = row["bbox"][3] - row["bbox"][1]
            super_height = supercell["bbox"][3] - supercell["bbox"][1]
            if row_height <= 0 or super_height <= 0:
                continue
            overlap = min(row["bbox"][3], supercell["bbox"][3]) - max(row["bbox"][1], supercell["bbox"][1])
            fraction = overlap / row_height
            if supercell.get("span"):
                fraction = max(fraction, overlap / super_height)
            if fraction >= 0.5:
                (header_rows if row.get("header") else data_rows).add(row_num)

        if header_rows and data_rows:
            if len(data_rows) > len(header_rows):
                header_rows = set()
            else:
                data_rows = set()
        if header_rows:
            supercell["header"] = True
        elif supercell.get("span"):
            continue

        intersecting_rows = data_rows.union(header_rows)
        if not intersecting_rows:
            continue

        intersecting_columns: list[int] = []
        for col_num, column in enumerate(columns):
            col_width = column["bbox"][2] - column["bbox"][0]
            super_width = supercell["bbox"][2] - supercell["bbox"][0]
            if col_width <= 0 or super_width <= 0:
                continue
            overlap = min(column["bbox"][2], supercell["bbox"][2]) - max(column["bbox"][0], supercell["bbox"][0])
            fraction = overlap / col_width
            if supercell.get("span"):
                fraction = max(fraction, overlap / super_width)
                if supercell["header"]:
                    fraction *= 2
            if fraction >= 0.5:
                intersecting_columns.append(col_num)

        if not intersecting_columns:
            continue
        if len(intersecting_rows) == 1 and len(intersecting_columns) == 1:
            continue

        row_boxes = [rows[idx]["bbox"] for idx in intersecting_rows]
        col_boxes = [columns[idx]["bbox"] for idx in intersecting_columns]
        row_rect = Rect()
        col_rect = Rect()
        for bbox in row_boxes:
            row_rect.include_rect(bbox)
        for bbox in col_boxes:
            col_rect.include_rect(bbox)

        supercell["bbox"] = list(row_rect.intersect(col_rect))
        supercell["row_numbers"] = sorted(intersecting_rows)
        supercell["column_numbers"] = intersecting_columns
        aligned.append(supercell)

    return aligned


def _remove_supercell_overlap(first: dict[str, Any], second: dict[str, Any]) -> None:
    common_rows = set(first["row_numbers"]).intersection(second["row_numbers"])
    common_columns = set(first["column_numbers"]).intersection(second["column_numbers"])

    while common_rows and common_columns:
        if len(second["row_numbers"]) < len(second["column_numbers"]):
            low = min(second["column_numbers"])
            high = max(second["column_numbers"])
            if high in common_columns:
                second["column_numbers"].remove(high)
                common_columns.remove(high)
            elif low in common_columns:
                second["column_numbers"].remove(low)
                common_columns.remove(low)
            else:
                second["column_numbers"] = []
                common_columns = set()
        else:
            low = min(second["row_numbers"])
            high = max(second["row_numbers"])
            if high in common_rows:
                second["row_numbers"].remove(high)
                common_rows.remove(high)
            elif low in common_rows:
                second["row_numbers"].remove(low)
                common_rows.remove(low)
            else:
                second["row_numbers"] = []
                common_rows = set()


def _nms_supercells(supercells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    supercells = sort_objects_by_score(supercells)
    suppressed = [False] * len(supercells)
    for second in range(1, len(supercells)):
        for first in range(second):
            _remove_supercell_overlap(supercells[first], supercells[second])
        if (
            (len(supercells[second]["row_numbers"]) < 2 and len(supercells[second]["column_numbers"]) < 2)
            or not supercells[second]["row_numbers"]
            or not supercells[second]["column_numbers"]
        ):
            suppressed[second] = True
    return [cell for idx, cell in enumerate(supercells) if not suppressed[idx]]


def _header_supercell_tree(supercells: list[dict[str, Any]]) -> None:
    header_cells = sort_objects_by_score([cell for cell in supercells if cell.get("header")])
    for cell in header_cells[:]:
        ancestors: defaultdict[int, int] = defaultdict(int)
        min_row = min(cell["row_numbers"])
        for other in header_cells:
            if max(other["row_numbers"]) < min_row and set(cell["column_numbers"]).issubset(other["column_numbers"]):
                for row in other["row_numbers"]:
                    ancestors[row] += 1
        for row in range(min_row):
            if ancestors[row] != 1:
                if cell in supercells:
                    supercells.remove(cell)
                break


def _refine_structure(
    structure: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    rows = structure["rows"]
    columns = structure["columns"]
    headers = [h for h in structure["headers"] if h["score"] >= thresholds["table column header"]]
    headers = _align_headers(_nms(headers), rows)

    supercells = [cell for cell in structure["supercells"] if not cell["subheader"]]
    subheaders = [cell for cell in structure["supercells"] if cell["subheader"]]
    supercells = [cell for cell in supercells if cell["score"] >= thresholds["table spanning cell"]]
    subheaders = [cell for cell in subheaders if cell["score"] >= thresholds["table projected row header"]]
    supercells = _align_supercells(supercells + subheaders, rows, columns)
    supercells = _nms_supercells(supercells)
    _header_supercell_tree(supercells)

    structure["headers"] = headers
    structure["supercells"] = supercells
    return structure


def objects_to_table_structures(
    table: dict[str, Any],
    objects: list[dict[str, Any]],
    tokens: list[dict[str, Any]],
    class_names: dict[int, str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    columns = [obj.copy() for obj in objects if class_names[obj["label"]] == "table column"]
    rows = [obj.copy() for obj in objects if class_names[obj["label"]] == "table row"]
    headers = [obj.copy() for obj in objects if class_names[obj["label"]] == "table column header"]
    supercells = [obj.copy() for obj in objects if class_names[obj["label"]] == "table spanning cell"]
    for cell in supercells:
        cell["subheader"] = False
    projected = [obj.copy() for obj in objects if class_names[obj["label"]] == "table projected row header"]
    for cell in projected:
        cell["subheader"] = True
    supercells.extend(projected)

    for row in rows:
        row["header"] = any(iob(row["bbox"], header["bbox"]) >= 0.5 for header in headers)
        row["page"] = table.get("page_num", 1)
    for column in columns:
        column["page"] = table.get("page_num", 1)

    rows = _refine_rows(rows, tokens)
    columns = _refine_columns(columns, tokens)
    if not rows or not columns:
        return {"rows": rows, "columns": columns, "headers": [], "supercells": []}

    row_rect = Rect()
    col_rect = Rect()
    for row in rows:
        row_rect.include_rect(row["bbox"])
    for column in columns:
        col_rect.include_rect(column["bbox"])
    table_bbox = [col_rect[0], row_rect[1], col_rect[2], row_rect[3]]
    table["bbox"] = table_bbox

    for column in columns:
        column["bbox"][1] = table_bbox[1]
        column["bbox"][3] = table_bbox[3]
    for row in rows:
        row["bbox"][0] = table_bbox[0]
        row["bbox"][2] = table_bbox[2]

    structure = {
        "rows": rows,
        "columns": columns,
        "headers": headers,
        "supercells": supercells,
    }
    if len(rows) > 0 and len(columns) > 1:
        structure = _refine_structure(structure, thresholds)
    return structure


def table_structure_to_cells(
    structure: dict[str, Any],
    tokens: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    columns = structure["columns"]
    rows = structure["rows"]
    supercells = structure["supercells"]
    cells: list[dict[str, Any]] = []
    subcells: list[dict[str, Any]] = []

    for col_num, column in enumerate(columns):
        for row_num, row in enumerate(rows):
            cell_rect = Rect(row["bbox"]).intersect(column["bbox"])
            if cell_rect.get_area() <= 0:
                continue
            cell = {
                "bbox": list(cell_rect),
                "column_nums": [col_num],
                "row_nums": [row_num],
                "header": bool(row.get("header")),
            }
            cell["subcell"] = any(
                Rect(supercell["bbox"]).intersect(cell_rect).get_area() / cell_rect.get_area() > 0.5
                for supercell in supercells
            )
            if cell["subcell"]:
                subcells.append(cell)
            else:
                cell["subheader"] = False
                cells.append(cell)

    for supercell in supercells:
        super_rect = Rect(supercell["bbox"])
        cell_columns: set[int] = set()
        cell_rows: set[int] = set()
        cell_rect: Rect | None = None
        header = True
        for subcell in subcells:
            sub_rect = Rect(subcell["bbox"])
            area = sub_rect.get_area()
            if area > 0 and sub_rect.intersect(super_rect).get_area() / area > 0.5:
                if cell_rect is None:
                    cell_rect = Rect(subcell["bbox"])
                else:
                    cell_rect.include_rect(subcell["bbox"])
                cell_rows.update(subcell["row_nums"])
                cell_columns.update(subcell["column_nums"])
                header = header and bool(subcell.get("header"))
        if cell_rect is not None and cell_rows and cell_columns:
            cells.append(
                {
                    "bbox": list(cell_rect),
                    "column_nums": sorted(cell_columns),
                    "row_nums": sorted(cell_rows),
                    "header": header,
                    "subheader": bool(supercell.get("subheader")),
                }
            )

    _, _, match_scores = slot_into_containers(cells, tokens)
    if match_scores:
        confidence = (sum(match_scores) / len(match_scores) + min(match_scores)) / 2
    else:
        confidence = 0.0

    token_nums_by_cell, _, _ = slot_into_containers(
        cells,
        tokens,
        overlap_threshold=0.001,
        unique_assignment=True,
        forced_assignment=False,
    )
    for cell, token_nums in zip(cells, token_nums_by_cell):
        cell_tokens = [tokens[num] for num in token_nums]
        cell["cell_text"] = extract_text_from_spans(cell_tokens)
        cell["spans"] = cell_tokens

    return cells, confidence


def objects_to_cells(
    table: dict[str, Any],
    objects: list[dict[str, Any]],
    tokens: list[dict[str, Any]],
    class_names: dict[int, str],
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    structure = objects_to_table_structures(table, objects, tokens, class_names, thresholds)
    if not structure["rows"] or not structure["columns"]:
        return structure, [], 0.0
    cells, confidence = table_structure_to_cells(structure, tokens)
    return structure, cells, confidence
