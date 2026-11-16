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
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

try:
    from rapidfuzz.distance.Levenshtein import distance as _fast_distance
except ImportError:
    _fast_distance = None


DEFAULT_TEXT_THRESHOLD = 0.5
DEFAULT_NUMBER_THRESHOLD = 0.1


@dataclasses.dataclass(frozen=True)
class RMSScores:
    """Relative Mapping Similarity precision, recall, and F1 on [0, 1]."""

    precision: float
    recall: float
    f1: float


@dataclasses.dataclass(frozen=True)
class MappingStatistics:
    """One selected mapping comparison, with scores on [0, 1]."""

    scores: RMSScores
    gold_count: int
    predicted_count: int
    matched_credit: float
    orientation: str


def mapping_scores(credit: float, gold_count: int, predicted_count: int) -> RMSScores:
    """Convert pooled mapping credit to PRF using the existing empty conventions."""
    if not gold_count and not predicted_count:
        return RMSScores(1.0, 1.0, 1.0)
    if not gold_count:
        return RMSScores(0.0, 1.0, 0.0)
    if not predicted_count:
        return RMSScores(1.0, 0.0, 0.0)
    if credit == 0:
        return RMSScores(0.0, 0.0, 0.0)
    precision, recall = credit / predicted_count, credit / gold_count
    return RMSScores(precision, recall, 2 * precision * recall / (precision + recall))


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
    if _fast_distance is not None:
        return _fast_distance(a, b)
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
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return rows_to_rms_text(csv.reader(handle))


def rows_to_rms_text(rows: Iterable[Sequence[str]]) -> str:
    """Apply the CSV metric's normalization to an in-memory table grid."""
    rows = [[normalize_cell(cell).lower() for cell in row] for row in rows]

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


def prepare_mappings(text: str, *, transposed: bool = False) -> dict[str, str]:
    """Prepare RMS mappings, retaining the historical duplicate-key behavior."""
    return _get_table_datapoints(_parse_table(text, transposed=transposed))


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
    return _mapping_credit(
        _get_table_datapoints(target_table), _get_table_datapoints(prediction_table),
        metric="rms", text_threshold=text_threshold, number_threshold=number_threshold,
        orientation="original",
    ).scores


def _mapping_credit(
    gold: dict[str, str], prediction: dict[str, str], *,
    metric: str, text_threshold: float, number_threshold: float, orientation: str,
) -> MappingStatistics:
    credit = 0.0
    if metric == "exact-mapping":
        credit = len(gold.items() & prediction.items())
    elif gold and prediction:
        expected, observed = list(gold.items()), list(prediction.items())
        # Cache repeated keys locally without retaining matrices across fragments.
        from functools import lru_cache
        key_similarity = lru_cache(maxsize=131072)(anls_metric)
        costs = [
            [1.0 - key_similarity(gkey, pkey, text_threshold) for pkey, _ in observed]
            for gkey, _ in expected
        ]
        row_indices, column_indices = _linear_sum_assignment(costs)
        for row, column in zip(row_indices, column_indices):
            credit += _get_datapoint_metric(
                expected[int(row)], observed[int(column)],
                text_threshold=text_threshold, number_threshold=number_threshold,
            )
    scores = mapping_scores(credit, len(gold), len(prediction))
    if metric == "exact-mapping" and gold and prediction:
        # Preserve Exact Mapping Match's original arithmetic and orientation ties.
        scores = dataclasses.replace(scores, f1=2 * credit / (len(gold) + len(prediction)))
    return MappingStatistics(
        scores,
        len(gold), len(prediction), credit, orientation,
    )


def evaluate_mapping_statistics(
    gold: dict[str, str], prediction_text: str, *, metric: str = "rms",
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    number_threshold: float = DEFAULT_NUMBER_THRESHOLD,
) -> MappingStatistics:
    """Score each orientation once and return its scores and sufficient statistics."""
    if metric not in {"rms", "exact-mapping"}:
        raise ValueError(f"Unsupported mapping metric: {metric!r}.")
    if metric == "rms" and not 0.0 <= text_threshold <= 1.0:
        raise ValueError("text_threshold must be between 0 and 1.")
    if metric == "rms" and not 0.0 <= number_threshold <= 1.0:
        raise ValueError("number_threshold must be between 0 and 1.")
    results = [
        _mapping_credit(
            gold, prepare_mappings(prediction_text, transposed=transposed),
            metric=metric, text_threshold=text_threshold, number_threshold=number_threshold,
            orientation="transposed" if transposed else "original",
        )
        for transposed in (True, False)
    ]
    return max(results, key=lambda result: result.scores.f1)


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

    return evaluate_mapping_statistics(
        prepare_mappings(target), prediction, metric="rms",
        text_threshold=text_threshold, number_threshold=number_threshold,
    ).scores


def exact_mapping_match(target: str, prediction: str) -> RMSScores:
    """Score exact normalized key/value pairs using RMS's mapping representation.

    Input text uses the same format as relative_mapping_similarity. CSV callers
    should use csv_to_rms_text first. Numeric values remain strings. Repeated
    keys retain the last value, matching the existing RMS dictionary semantics.
    Empty mappings and orientation ties follow RMS's conventions.
    """
    return evaluate_mapping_statistics(
        prepare_mappings(target), prediction, metric="exact-mapping",
    ).scores
