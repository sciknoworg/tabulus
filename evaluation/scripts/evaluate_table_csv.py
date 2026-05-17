from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from evaluation.deplot import metrics


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def evaluate(ground_truth_csv: Path, prediction_csv: Path) -> Dict[str, Any]:
    scores = metrics.table_datapoints_precision_recall(
        target=str(ground_truth_csv),
        prediction=str(prediction_csv),
    )

    result = {
        "ground_truth_csv": str(ground_truth_csv),
        "prediction_csv": str(prediction_csv),
        "precision": scores.get("precision"),
        "recall": scores.get("recall"),
        "f1": scores.get("f1"),
    }

    return result


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage:")
        print("python evaluate_table_csv.py <ground_truth.csv> <prediction.csv>")
        raise SystemExit(1)

    ground_truth_csv = Path(sys.argv[1]).resolve()
    prediction_csv = Path(sys.argv[2]).resolve()

    if not ground_truth_csv.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {ground_truth_csv}")

    if not prediction_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_csv}")

    result = evaluate(ground_truth_csv, prediction_csv)

    evaluation_dir = prediction_csv.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    output_json = evaluation_dir / f"{prediction_csv.stem}_evaluation.json"

    write_json(output_json, result)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()