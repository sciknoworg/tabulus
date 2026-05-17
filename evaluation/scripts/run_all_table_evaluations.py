from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_PREDICTION_DIRS = [
    "deepseek2_prediction",
    "paddle_vl_prediction",
    "chandra_prediction",
    "Kreuzberg_prediction",
]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def find_ground_truth_dirs(dataset_root: Path) -> List[Path]:
    return sorted(
        path
        for path in dataset_root.rglob(".csv")
        if path.is_dir() and path.parent.name == "Ref_Tables"
    )


def run_evaluation(
    dataset_root: Path,
    evaluation_script: Path,
    prediction_dirs: List[str],
) -> Dict[str, Any]:
    success = []
    failed = []
    missing = []

    gt_dirs = find_ground_truth_dirs(dataset_root)

    print(f"Found {len(gt_dirs)} ground-truth folders.\n", flush=True)

    for gt_dir in gt_dirs:
        ref_tables_dir = gt_dir.parent
        gt_files = sorted(gt_dir.glob("*.csv"))

        if not gt_files:
            continue

        print(f"\nGround truth folder: {gt_dir}", flush=True)
        print(f"Found {len(gt_files)} ground-truth CSV files", flush=True)

        for pred_dir_name in prediction_dirs:
            pred_dir = ref_tables_dir / pred_dir_name

            if not pred_dir.exists():
                print(f"  Prediction folder missing: {pred_dir}", flush=True)
                missing.append(str(pred_dir))
                continue

            print(f"  Evaluating predictions in: {pred_dir}", flush=True)

            for gt_file in gt_files:
                pred_file = pred_dir / gt_file.name

                if not pred_file.exists():
                    print(f"    Missing prediction for: {gt_file.name}", flush=True)
                    missing.append(str(pred_file))
                    continue

                print(f"    Comparing: {gt_file.name}", flush=True)

                result = subprocess.run(
                    [
                        sys.executable,
                        str(evaluation_script),
                        str(gt_file),
                        str(pred_file),
                    ],
                    text=True,
                    capture_output=True,
                )

                record = {
                    "ground_truth": str(gt_file),
                    "prediction": str(pred_file),
                    "prediction_model": pred_dir_name,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

                if result.returncode == 0:
                    print("      -> Done", flush=True)
                    success.append(record)
                else:
                    print(f"      -> Failed with exit code {result.returncode}", flush=True)
                    failed.append(record)

    summary = {
        "dataset_root": str(dataset_root),
        "evaluation_script": str(evaluation_script),
        "prediction_dirs": prediction_dirs,
        "successful_evaluations": len(success),
        "failed_evaluations": len(failed),
        "missing_files_or_folders": len(missing),
        "success": success,
        "failed": failed,
        "missing": missing,
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run table extraction evaluation for all prediction folders."
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to dataset root folder.",
    )

    parser.add_argument(
        "--evaluation-script",
        default="./evaluation/evaluate_table_csv.py",
        help="Path to single-table evaluation script.",
    )

    parser.add_argument(
        "--prediction-dirs",
        nargs="+",
        default=DEFAULT_PREDICTION_DIRS,
        help="Prediction folder names inside Ref_Tables.",
    )

    parser.add_argument(
        "--summary-out",
        default="./evaluation/table_evaluation_summary.json",
        help="Path to save summary JSON.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    evaluation_script = Path(args.evaluation_script).resolve()
    summary_out = Path(args.summary_out).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if not evaluation_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {evaluation_script}")

    summary = run_evaluation(
        dataset_root=dataset_root,
        evaluation_script=evaluation_script,
        prediction_dirs=args.prediction_dirs,
    )

    write_json(summary_out, summary)

    print("\nFinished.", flush=True)
    print(f"Successful evaluations: {summary['successful_evaluations']}", flush=True)
    print(f"Failed evaluations: {summary['failed_evaluations']}", flush=True)
    print(f"Missing files/folders: {summary['missing_files_or_folders']}", flush=True)
    print(f"Summary written to: {summary_out}", flush=True)


if __name__ == "__main__":
    main()