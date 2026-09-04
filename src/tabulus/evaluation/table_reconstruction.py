"""Generic table-reconstruction evaluation for Tabulus prediction CSVs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tabulus.evaluation.rms import (
    DEFAULT_NUMBER_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    csv_to_rms_text,
    relative_mapping_similarity,
)


SUPPORTED_TABLE_RECONSTRUCTION_METRICS = ("rms",)


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
    text_threshold: float
    number_threshold: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_csv"] = str(self.gold_csv)
        payload["prediction_csv"] = str(self.prediction_csv)
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

    if metric == "rms":
        scores = relative_mapping_similarity(
            csv_to_rms_text(gold_path),
            csv_to_rms_text(prediction_path),
            text_threshold=text_threshold,
            number_threshold=number_threshold,
        )
    else:  # pragma: no cover - guarded by metric validation above
        raise AssertionError(f"Unhandled table reconstruction metric: {metric}")

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
