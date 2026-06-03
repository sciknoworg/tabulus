from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import optimize


GOLD_DIR_NAME = "gold_standard.csv"

PRED_DIR_NAMES = [
    "deepseek2_prediction",
    "paddle_vl_prediction",
    "chandra_prediction",
    "Kreuzberg_prediction",
    "NuExtract3_prediction",
]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        current = [i]

        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )

        previous = current

    return previous[-1]


def anls_metric(target: str, prediction: str, theta: float = 0.5) -> float:
    target = str(target).lower().strip()
    prediction = str(prediction).lower().strip()

    if not target and not prediction:
        return 1.0

    if not target or not prediction:
        return 0.0

    distance = levenshtein(target, prediction) / max(len(target), len(prediction))

    return 1.0 - distance if distance < theta else 0.0


def normalize_cell(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


def csv_to_deplot_text(path: Path) -> str:
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            rows.append([normalize_cell(cell) for cell in row])

    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]

    return "\n".join(" | ".join(row) for row in rows)


def _to_float(text: str):
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0

        return float(text)

    except ValueError:
        return None


def _get_relative_distance(target, prediction, theta=1.0):
    if not target:
        return int(not prediction)

    distance = min(abs((target - prediction) / target), 1)

    return distance if distance < theta else 1


def _permute(values, indexes):
    return tuple(values[i] if i < len(values) else "" for i in indexes)


@dataclasses.dataclass(frozen=True)
class Table:
    title: Optional[str] = None
    headers: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    rows: tuple[tuple[str, ...], ...] = dataclasses.field(default_factory=tuple)

    def permuted(self, indexes):
        return Table(
            title=self.title,
            headers=_permute(self.headers, indexes),
            rows=tuple(_permute(row, indexes) for row in self.rows),
        )


def _parse_table(text: str, transposed=False) -> Table:
    lines = text.lower().splitlines()

    if not lines:
        return Table()

    if lines[0].startswith("title |"):
        title = lines[0][len("title |") :].strip()
        offset = 1
    else:
        title = None
        offset = 0

    if len(lines) < offset + 1:
        return Table(title=title)

    rows = []

    for line in lines[offset:]:
        rows.append(tuple(v.strip() for v in line.split(" | ")))

    if transposed:
        rows = [tuple(row) for row in itertools.zip_longest(*rows, fillvalue="")]

    return Table(
        title=title,
        headers=rows[0],
        rows=tuple(rows[1:]),
    )


def _get_table_datapoints(table: Table):
    datapoints = {}

    if table.title is not None:
        datapoints["title"] = table.title

    if not table.rows or len(table.headers) <= 1:
        return datapoints

    for row in table.rows:
        for header, cell in zip(table.headers[1:], row[1:]):
            datapoints[f"{row[0]} {header}"] = cell

    return datapoints


def _get_datapoint_metric(
    target,
    prediction,
    text_theta=0.5,
    number_theta=0.1,
):
    key_metric = anls_metric(target[0], prediction[0], text_theta)

    pred_float = _to_float(prediction[1])
    target_float = _to_float(target[1])

    if pred_float is not None and target_float:
        return key_metric * (
            1 - _get_relative_distance(target_float, pred_float, number_theta)
        )

    if target[1] == prediction[1]:
        return key_metric

    return key_metric * anls_metric(target[1], prediction[1], text_theta)


def _table_datapoints_precision_recall_f1(
    target_table: Table,
    prediction_table: Table,
    text_theta=0.5,
    number_theta=0.1,
):
    target_datapoints = list(_get_table_datapoints(target_table).items())
    prediction_datapoints = list(_get_table_datapoints(prediction_table).items())

    if not target_datapoints and not prediction_datapoints:
        return 1.0, 1.0, 1.0

    if not target_datapoints:
        return 0.0, 1.0, 0.0

    if not prediction_datapoints:
        return 1.0, 0.0, 0.0

    distance = []

    for t, _ in target_datapoints:
        distance.append(
            [1 - anls_metric(t, p, text_theta) for p, _ in prediction_datapoints]
        )

    cost_matrix = np.array(distance)
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)

    score = 0.0

    for r, c in zip(row_ind, col_ind):
        score += _get_datapoint_metric(
            target_datapoints[r],
            prediction_datapoints[c],
            text_theta,
            number_theta,
        )

    if score == 0:
        return 0.0, 0.0, 0.0

    precision = score / len(prediction_datapoints)
    recall = score / len(target_datapoints)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def rms_precision_recall_f1(target: str, prediction: str):
    all_metrics = []

    for transposed in [True, False]:
        pred_table = _parse_table(prediction, transposed=transposed)

        all_metrics.append(
            _table_datapoints_precision_recall_f1(
                _parse_table(target),
                pred_table,
            )
        )

    precision, recall, f1 = max(all_metrics, key=lambda x: x[-1])

    return {
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "rms_f1": 100.0 * f1,
    }


def evaluate_pair(gold_csv: Path, pred_csv: Path, model_name: str):
    target_text = csv_to_deplot_text(gold_csv)
    prediction_text = csv_to_deplot_text(pred_csv)

    scores = rms_precision_recall_f1(
        target=target_text,
        prediction=prediction_text,
    )

    return {
        "ground_truth_csv": str(gold_csv),
        "prediction_csv": str(pred_csv),
        "prediction_model": model_name,
        "precision": scores["precision"],
        "recall": scores["recall"],
        "rms_f1": scores["rms_f1"],
    }


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def find_gold_dirs(dataset_root: Path):
    return sorted(
        path
        for path in dataset_root.rglob(GOLD_DIR_NAME)
        if path.is_dir() and path.parent.name == "Ref_Tables"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate precision, recall and RMS F1 for all prediction folders."
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to dataset root.",
    )

    parser.add_argument(
        "--summary-out",
        default="all_models_rms_precision_recall_f1_summary.json",
        help="Output summary JSON.",
    )

    parser.add_argument(
        "--prediction-dirs",
        nargs="+",
        default=PRED_DIR_NAMES,
        help="Prediction folder names inside Ref_Tables.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    summary_out = Path(args.summary_out).resolve()

    results = []
    failed = []
    missing = []

    gold_dirs = find_gold_dirs(dataset_root)

    print(f"Found gold folders: {len(gold_dirs)}", flush=True)

    for gold_dir in gold_dirs:
        ref_tables_dir = gold_dir.parent

        for model_name in args.prediction_dirs:
            pred_dir = ref_tables_dir / model_name

            if not pred_dir.exists():
                missing.append(str(pred_dir))
                continue

            output_dir = ref_tables_dir / "evaluation" / model_name
            output_dir.mkdir(parents=True, exist_ok=True)

            for gold_csv in sorted(gold_dir.glob("*.csv")):
                pred_csv = pred_dir / gold_csv.name

                if not pred_csv.exists():
                    missing.append(str(pred_csv))
                    continue

                try:
                    result = evaluate_pair(
                        gold_csv=gold_csv,
                        pred_csv=pred_csv,
                        model_name=model_name,
                    )

                    results.append(result)

                    out_json = output_dir / f"{gold_csv.stem}_{model_name}_rms_eval.json"
                    write_json(out_json, result)

                    print(
                        f"{model_name} / {gold_csv.name}: "
                        f"P={result['precision']:.4f}, "
                        f"R={result['recall']:.4f}, "
                        f"F1={result['rms_f1']:.4f}",
                        flush=True,
                    )

                except Exception as e:
                    failed.append(
                        {
                            "model": model_name,
                            "ground_truth_csv": str(gold_csv),
                            "prediction_csv": str(pred_csv),
                            "error": str(e),
                        }
                    )

                    print(
                        f"Failed: {model_name} / {gold_csv.name} -> {e}",
                        flush=True,
                    )

    model_summaries = {}

    for model_name in args.prediction_dirs:
        model_results = [
            r for r in results if r["prediction_model"] == model_name
        ]

        n = len(model_results)

        model_summaries[model_name] = {
            "successful_evaluations": n,
            "average_precision": (
                sum(float(r["precision"]) for r in model_results) / n
                if n
                else 0.0
            ),
            "average_recall": (
                sum(float(r["recall"]) for r in model_results) / n
                if n
                else 0.0
            ),
            "average_rms_f1": (
                sum(float(r["rms_f1"]) for r in model_results) / n
                if n
                else 0.0
            ),
        }

    summary = {
        "dataset_root": str(dataset_root),
        "prediction_dirs": args.prediction_dirs,
        "total_successful_evaluations": len(results),
        "failed_evaluations": len(failed),
        "missing_files_or_folders": len(missing),
        "model_summaries": model_summaries,
        "results": results,
        "failed": failed,
        "missing": missing,
    }

    write_json(summary_out, summary)

    print("\nFinished.", flush=True)

    for model_name, data in model_summaries.items():
        print(
            f"{model_name}: "
            f"n={data['successful_evaluations']}, "
            f"P={data['average_precision']:.4f}, "
            f"R={data['average_recall']:.4f}, "
            f"F1={data['average_rms_f1']:.4f}",
            flush=True,
        )

    print(f"\nSummary written to: {summary_out}", flush=True)


if __name__ == "__main__":
    main()