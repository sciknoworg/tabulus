# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Relative Mapping Similarity (RMS) for structured table comparison.

This module adapts the table-datapoint metric released with Google Research
DePlot for use as a small, file-format-independent Tabulus evaluation primitive.
Scores are returned on the natural [0, 1] scale used by the per-example DePlot
implementation.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import re
from pathlib import Path
from typing import Any


DEFAULT_TEXT_THRESHOLD = 0.5
DEFAULT_NUMBER_THRESHOLD = 0.1


@dataclasses.dataclass(frozen=True)
class RMSScores:
    """Relative Mapping Similarity precision, recall, and F1 on [0, 1]."""

    precision: float
    recall: float
    f1: float


def _linear_sum_assignment(cost_matrix: list[list[float]]):
    """Load SciPy lazily so non-evaluation Tabulus commands stay lightweight."""

    try:
        from scipy import optimize
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Relative Mapping Similarity evaluation requires the optional "
            "Tabulus evaluation dependencies. Install them with "
            "`pip install 'tabulus[evaluation]'`."
        ) from error

    return optimize.linear_sum_assignment(cost_matrix)


def levenshtein(a: str, b: str) -> int:
    """Return Levenshtein edit distance without an additional dependency."""

    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current

    return previous[-1]


def anls_metric(target: str, prediction: str, theta: float = 0.5) -> float:
    """Return Average Normalized Levenshtein Similarity for one string pair."""

    target = str(target).lower().strip()
    prediction = str(prediction).lower().strip()

    if not target and not prediction:
        return 1.0
    if not target or not prediction:
        return 0.0

    distance = levenshtein(target, prediction) / max(len(target), len(prediction))
    return 1.0 - distance if distance < theta else 0.0


def normalize_cell(value: Any) -> str:
    """Normalize CSV cell text before converting it to DePlot table syntax."""

    text = str(value or "")
    text = text.replace("\ufeff", "")
    return re.sub(r"\s+", " ", text.strip())


def csv_to_rms_text(path: Path) -> str:
    """Convert a CSV file to the flattened table text expected by RMS."""

    path = Path(path)
    rows: list[list[str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows.append([normalize_cell(cell).lower() for cell in row])

    if not rows:
        return ""

    max_columns = max(len(row) for row in rows)
    normalized_rows = [
        row + [""] * (max_columns - len(row))
        for row in rows
    ]

    return "\n".join(" | ".join(row) for row in normalized_rows)


def _to_float(text: str) -> float | None:
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except ValueError:
        return None


def _get_relative_distance(
    target: float,
    prediction: float,
    theta: float = 1.0,
) -> float:
    if not target:
        return float(int(not prediction))

    distance = min(abs((target - prediction) / target), 1.0)
    return distance if distance < theta else 1.0


def _permute(values: tuple[str, ...], indexes: list[int]) -> tuple[str, ...]:
    return tuple(values[index] if index < len(values) else "" for index in indexes)


@dataclasses.dataclass(frozen=True)
class Table:
    """Minimal table representation used by the DePlot RMS computation."""

    title: str | None = None
    headers: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    rows: tuple[tuple[str, ...], ...] = dataclasses.field(default_factory=tuple)

    def permuted(self, indexes: list[int]) -> "Table":
        return Table(
            title=self.title,
            headers=_permute(self.headers, indexes),
            rows=tuple(_permute(row, indexes) for row in self.rows),
        )


def _parse_table(text: str, *, transposed: bool = False) -> Table:
    lines = text.lower().splitlines()

    if not lines:
        return Table()

    if lines[0].startswith("title |"):
        title = lines[0][len("title |"):].strip()
        offset = 1
    else:
        title = None
        offset = 0

    if len(lines) < offset + 1:
        return Table(title=title)

    rows = [tuple(value.strip() for value in line.split(" | ")) for line in lines[offset:]]

    if transposed:
        rows = [tuple(row) for row in itertools.zip_longest(*rows, fillvalue="")]

    return Table(
        title=title,
        headers=rows[0],
        rows=tuple(rows[1:]),
    )


def _get_table_datapoints(table: Table) -> dict[str, str]:
    datapoints: dict[str, str] = {}

    if table.title is not None:
        datapoints["title"] = table.title

    if not table.rows or len(table.headers) <= 1:
        return datapoints

    for row in table.rows:
        for header, cell in zip(table.headers[1:], row[1:]):
            datapoints[f"{row[0]} {header}"] = cell

    return datapoints


def _get_datapoint_metric(
    target: tuple[str, str],
    prediction: tuple[str, str],
    *,
    text_threshold: float,
    number_threshold: float,
) -> float:
    key_metric = anls_metric(target[0], prediction[0], text_threshold)

    prediction_float = _to_float(prediction[1])
    target_float = _to_float(target[1])

    # Preserve DePlot's original truthiness check for the target numeric value.
    if prediction_float is not None and target_float:
        return key_metric * (
            1.0
            - _get_relative_distance(
                target_float,
                prediction_float,
                number_threshold,
            )
        )

    if target[1] == prediction[1]:
        return key_metric

    return key_metric * anls_metric(
        target[1],
        prediction[1],
        text_threshold,
    )


def _table_datapoints_precision_recall_f1(
    target_table: Table,
    prediction_table: Table,
    *,
    text_threshold: float,
    number_threshold: float,
) -> RMSScores:
    target_datapoints = list(_get_table_datapoints(target_table).items())
    prediction_datapoints = list(_get_table_datapoints(prediction_table).items())

    if not target_datapoints and not prediction_datapoints:
        return RMSScores(precision=1.0, recall=1.0, f1=1.0)
    if not target_datapoints:
        return RMSScores(precision=0.0, recall=1.0, f1=0.0)
    if not prediction_datapoints:
        return RMSScores(precision=1.0, recall=0.0, f1=0.0)

    cost_matrix = [
        [
            1.0 - anls_metric(target_key, prediction_key, text_threshold)
            for prediction_key, _ in prediction_datapoints
        ]
        for target_key, _ in target_datapoints
    ]
    row_indices, column_indices = _linear_sum_assignment(cost_matrix)

    score = 0.0
    for row_index, column_index in zip(row_indices, column_indices):
        score += _get_datapoint_metric(
            target_datapoints[int(row_index)],
            prediction_datapoints[int(column_index)],
            text_threshold=text_threshold,
            number_threshold=number_threshold,
        )

    if score == 0.0:
        return RMSScores(precision=0.0, recall=0.0, f1=0.0)

    precision = score / len(prediction_datapoints)
    recall = score / len(target_datapoints)
    f1 = 2.0 * precision * recall / (precision + recall)

    return RMSScores(precision=precision, recall=recall, f1=f1)


def relative_mapping_similarity(
    target: str,
    prediction: str,
    *,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    number_threshold: float = DEFAULT_NUMBER_THRESHOLD,
) -> RMSScores:
    """Compute DePlot Relative Mapping Similarity for one table pair.

    The prediction is evaluated in both its original and transposed orientation,
    matching the DePlot table-datapoint evaluation behavior. The orientation
    with the highest F1 is retained.
    """

    if not 0.0 <= text_threshold <= 1.0:
        raise ValueError("text_threshold must be between 0 and 1.")
    if not 0.0 <= number_threshold <= 1.0:
        raise ValueError("number_threshold must be between 0 and 1.")

    target_table = _parse_table(target)
    scores = [
        _table_datapoints_precision_recall_f1(
            target_table,
            _parse_table(prediction, transposed=transposed),
            text_threshold=text_threshold,
            number_threshold=number_threshold,
        )
        for transposed in (True, False)
    ]

    return max(scores, key=lambda item: item.f1)
