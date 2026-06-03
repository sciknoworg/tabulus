from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


CLASS_COLUMNS = [
    "Grid",
    "Column hierarchy",
    "Row hierarchy",
    "Cell density",
    "size",
    "sections",

]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_table_name(name: str) -> str:
    name = Path(str(name)).name
    name = name.replace(".png", "").replace(".csv", "")
    return name.strip().lower()


def extract_paper_id(path_text: str) -> str | None:
    match = re.search(r"(P\d+)", str(path_text).replace("\\", "/"))
    return match.group(1) if match else None


def make_key(paper_id: str | None, table_name: str) -> str:
    return f"{paper_id or ''}::{normalize_table_name(table_name)}"


def load_classification_csv(path: Path) -> Dict[str, Dict[str, str]]:
    rows_by_key = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            paper_id = extract_paper_id(row.get("path", ""))
            table_name = row.get("table", "")

            key = make_key(paper_id, table_name)
            rows_by_key[key] = row

    return rows_by_key


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate RMS precision, recall and F1 by table subclass."
    )

    parser.add_argument(
        "--scores-json",
        required=True,
        help="Path to all_models_rms_precision_recall_f1_summary.json.",
    )

    parser.add_argument(
        "--classification-csv",
        required=True,
        help="Path to table subclass classification CSV.",
    )

    parser.add_argument(
        "--output-json",
        default="subclass_scores_summary.json",
        help="Output JSON path.",
    )

    args = parser.parse_args()

    scores_json = Path(args.scores_json).resolve()
    classification_csv = Path(args.classification_csv).resolve()
    output_json = Path(args.output_json).resolve()

    scores_data = read_json(scores_json)
    class_rows = load_classification_csv(classification_csv)

    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    unmatched = []

    for result in scores_data.get("results", []):
        model = result.get("prediction_model")
        gold_path = result.get("ground_truth_csv", "")

        paper_id = extract_paper_id(gold_path)
        table_name = Path(gold_path).stem
        key = make_key(paper_id, table_name)

        class_row = class_rows.get(key)

        if not class_row:
            unmatched.append(result)
            continue

        for column in CLASS_COLUMNS:
            class_value = str(class_row.get(column, "")).strip() or "unknown"

            grouped.setdefault(model, {})
            grouped[model].setdefault(column, {})
            grouped[model][column].setdefault(class_value, {
                "precision": [],
                "recall": [],
                "rms_f1": [],
            })

            grouped[model][column][class_value]["precision"].append(
                safe_float(result.get("precision"))
            )
            grouped[model][column][class_value]["recall"].append(
                safe_float(result.get("recall"))
            )
            grouped[model][column][class_value]["rms_f1"].append(
                safe_float(result.get("rms_f1"))
            )

    summary = {
        "scores_json": str(scores_json),
        "classification_csv": str(classification_csv),
        "matched_results": 0,
        "unmatched_results": len(unmatched),
        "subclass_scores": {},
        "unmatched_examples": unmatched[:20],
    }

    matched_count = 0

    for model, model_data in grouped.items():
        summary["subclass_scores"][model] = {}

        for column, column_data in model_data.items():
            summary["subclass_scores"][model][column] = {}

            for class_value, metrics in column_data.items():
                count = len(metrics["rms_f1"])
                matched_count += count

                summary["subclass_scores"][model][column][class_value] = {
                    "count": count,
                    "average_precision": mean(metrics["precision"]),
                    "average_recall": mean(metrics["recall"]),
                    "average_rms_f1": mean(metrics["rms_f1"]),
                }

    summary["matched_results"] = matched_count

    write_json(output_json, summary)

    print("Finished.")
    print(f"Matched results: {matched_count}")
    print(f"Unmatched results: {len(unmatched)}")
    print(f"Output written to: {output_json}")


if __name__ == "__main__":
    main()