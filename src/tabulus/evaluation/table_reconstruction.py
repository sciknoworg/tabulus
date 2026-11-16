"""Generic table-reconstruction evaluation for Tabulus prediction CSVs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tabulus.evaluation.rms import (
    DEFAULT_NUMBER_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,

)

from tabulus.evaluation.grid_scoring import ExactCellCounts, count_exact_cells, score_table_grids
from tabulus.resolved_export.csv_io import read_csv


SUPPORTED_TABLE_RECONSTRUCTION_METRICS = ("rms", "exact", "exact-mapping")


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
    statistics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_csv"] = str(self.gold_csv)
        payload["prediction_csv"] = str(self.prediction_csv)
        if self.statistics is None:
            del payload["statistics"]
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

    scored = score_table_grids(
        read_csv(gold_path), read_csv(prediction_path), metrics=(metric,),
        text_threshold=text_threshold, number_threshold=number_threshold,
    )
    names = {
        "rms": ("Relative Mapping Similarity", "RMS", "DePlot"),
        "exact": ("Exact Cell Match", "Exact", "Position-based CSV comparison"),
        "exact-mapping": ("Exact Mapping Match", "EM", "DePlot mapping representation with exact key/value equality"),
    }
    name, short_name, implementation = names[metric]
    return TableReconstructionEvaluation(
        metric=metric, metric_name=name, metric_short_name=short_name,
        implementation=implementation, score_scale="[0,100]",
        gold_csv=gold_path, prediction_csv=prediction_path,
        text_threshold=text_threshold if metric == "rms" else None,
        number_threshold=number_threshold if metric == "rms" else None,
        **scored["metrics"][metric],
        exact_cells=ExactCellCounts(**scored["exact_cells"]) if metric == "exact" else None,
        statistics=scored["statistics"][metric],
    )
