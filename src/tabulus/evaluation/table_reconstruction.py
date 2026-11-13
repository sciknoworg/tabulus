"""Generic table-reconstruction evaluation for Tabulus prediction CSVs."""

from __future__ import annotations

import csv
import json
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tabulus.evaluation.rms import (
    DEFAULT_NUMBER_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    csv_to_rms_text,
    exact_mapping_match,
    relative_mapping_similarity,
)


SUPPORTED_TABLE_RECONSTRUCTION_METRICS = ("rms", "exact", "exact-mapping")


@dataclass(frozen=True)
class ExactCellCounts:
    """Nonempty cell counts and whole-grid equality after text normalization."""

    matching_cells: int
    gold_nonempty_cells: int
    predicted_nonempty_cells: int
    exact_grid: bool


def count_exact_cells(
    gold: Iterable[Sequence[str]],
    prediction: Iterable[Sequence[str]],
) -> ExactCellCounts:
    """Count exact nonempty matches at identical row and column positions."""
    def normalize_grid(rows: Iterable[Sequence[str]]) -> list[list[str]]:
        return [
            [" ".join(unicodedata.normalize("NFC", cell).split()) for cell in row]
            for row in rows
        ]

    gold_grid = normalize_grid(gold)
    prediction_grid = normalize_grid(prediction)
    return ExactCellCounts(
        matching_cells=sum(
            bool(expected) and expected == actual
            for gold_row, prediction_row in zip(gold_grid, prediction_grid)
            for expected, actual in zip(gold_row, prediction_row)
        ),
        gold_nonempty_cells=sum(bool(cell) for row in gold_grid for cell in row),
        predicted_nonempty_cells=sum(bool(cell) for row in prediction_grid for cell in row),
        exact_grid=gold_grid == prediction_grid,
    )


@dataclass(frozen=True)
class TableReconstructionEvaluation:
    """Evaluation result for one gold/prediction table pair."""

    metric: str
    metric_name: str
    metric_short_name: str
    implementation: str
    score_scale: str
    gold_csv: Path
    prediction_csv: Path
    text_threshold: float | None
    number_threshold: float | None
    precision: float
    recall: float
    f1: float
    exact_cells: ExactCellCounts | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_csv"] = str(self.gold_csv)
        payload["prediction_csv"] = str(self.prediction_csv)
        if self.exact_cells is None:
            del payload["exact_cells"]
        if self.exact_cells is not None or self.metric == "exact-mapping":
            del payload["text_threshold"]
            del payload["number_threshold"]
        return payload

    def write_json(self, path: Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path


def _require_csv(path: Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {resolved}")
    if resolved.suffix.lower() != ".csv":
        raise ValueError(f"{label} must be a CSV file: {resolved}")

    return resolved


def evaluate_table_reconstruction(
    gold_csv: Path,
    prediction_csv: Path,
    *,
    metric: str = "rms",
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    number_threshold: float = DEFAULT_NUMBER_THRESHOLD,
) -> TableReconstructionEvaluation:
    """Evaluate one reconstructed table against one gold-standard CSV.

    This API is intentionally dataset-agnostic. It consumes only an explicit
    gold CSV and prediction CSV and does not depend on TabulusBench layout,
    paper identifiers, or benchmark metadata.
    """

    if metric not in SUPPORTED_TABLE_RECONSTRUCTION_METRICS:
        raise ValueError(
            f"Unsupported table reconstruction metric: {metric!r}. "
            "Supported metrics: "
            + ", ".join(SUPPORTED_TABLE_RECONSTRUCTION_METRICS)
        )

    gold_path = _require_csv(gold_csv, label="Gold")
    prediction_path = _require_csv(prediction_csv, label="Prediction")

    if metric == "exact":
        with (
            gold_path.open(encoding="utf-8-sig", newline="") as gold_file,
            prediction_path.open(encoding="utf-8-sig", newline="") as prediction_file,
        ):
            counts = count_exact_cells(csv.reader(gold_file), csv.reader(prediction_file))

        return TableReconstructionEvaluation(
            metric="exact",
            metric_name="Exact Cell Match",
            metric_short_name="Exact",
            implementation="Position-based CSV comparison",
            score_scale="[0,100]",
            gold_csv=gold_path,
            prediction_csv=prediction_path,
            text_threshold=None,
            number_threshold=None,
            precision=100.0 * counts.matching_cells / max(1, counts.predicted_nonempty_cells),
            recall=100.0 * counts.matching_cells / max(1, counts.gold_nonempty_cells),
            f1=200.0 * counts.matching_cells / max(
                1, counts.gold_nonempty_cells + counts.predicted_nonempty_cells
            ),
            exact_cells=counts,
        )

    if metric == "exact-mapping":
        scores = exact_mapping_match(csv_to_rms_text(gold_path), csv_to_rms_text(prediction_path))
        return TableReconstructionEvaluation(
            metric="exact-mapping",
            metric_name="Exact Mapping Match",
            metric_short_name="EM",
            implementation="DePlot mapping representation with exact key/value equality",
            score_scale="[0,100]",
            gold_csv=gold_path,
            prediction_csv=prediction_path,
            text_threshold=None,
            number_threshold=None,
            precision=100.0 * scores.precision,
            recall=100.0 * scores.recall,
            f1=100.0 * scores.f1,
        )

    scores = relative_mapping_similarity(
        csv_to_rms_text(gold_path),
        csv_to_rms_text(prediction_path),
        text_threshold=text_threshold,
        number_threshold=number_threshold,
    )

    return TableReconstructionEvaluation(
        metric="rms",
        metric_name="Relative Mapping Similarity",
        metric_short_name="RMS",
        implementation="DePlot",
        score_scale="[0,100]",
        gold_csv=gold_path,
        prediction_csv=prediction_path,
        text_threshold=text_threshold,
        number_threshold=number_threshold,
        precision=100.0 * scores.precision,
        recall=100.0 * scores.recall,
        f1=100.0 * scores.f1,
    )
