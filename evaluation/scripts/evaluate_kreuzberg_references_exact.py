from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Set, Tuple


GOLD_FILENAME = "gold_standard_references.json"
DEFAULT_PRED_FILENAME = "extracted_kreuzberg_ref.json"
DEFAULT_OUTPUT_FILENAME = "kreuzberg_plus_regex_f1_score_result.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_ref(ref: str) -> str:
    return " ".join(ref.strip().split())


def to_reference_set(data: Any) -> Set[Tuple[str, str]]:
    if not isinstance(data, list):
        return set()

    return {
        (
            str(item.get("nr", "")).strip(),
            normalize_ref(str(item.get("ref", ""))),
        )
        for item in data
        if isinstance(item, dict) and "nr" in item and "ref" in item
    }


def calculate_f1(gold_data: Any, pred_data: Any) -> Dict[str, Any]:
    gold_set = to_reference_set(gold_data)
    pred_set = to_reference_set(pred_data)

    true_positives = len(gold_set & pred_set)
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "gold_total": len(gold_set),
        "prediction_total": len(pred_set),
    }


def process_ref_folder(
    ref_folder: Path,
    pred_filename: str,
    output_filename: str,
) -> bool:
    gold_path = ref_folder / GOLD_FILENAME
    pred_path = ref_folder / pred_filename
    output_path = ref_folder / output_filename

    if not gold_path.exists():
        print(f"Skipped: {ref_folder} -> missing {GOLD_FILENAME}", flush=True)
        return False

    if not pred_path.exists():
        print(f"Skipped: {ref_folder} -> missing {pred_filename}", flush=True)
        return False

    gold_data = load_json(gold_path)
    pred_data = load_json(pred_path)

    result = calculate_f1(gold_data, pred_data)

    result["input_files"] = {
        "gold": str(gold_path),
        "prediction": str(pred_path),
    }

    write_json(output_path, result)

    print(f"Created: {output_path}", flush=True)
    print(f"F1: {result['f1_score']:.3f}", flush=True)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Kreuzberg + regex reference extraction using exact nr/ref matching."
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to dataset root.",
    )

    parser.add_argument(
        "--pred-filename",
        default=DEFAULT_PRED_FILENAME,
        help="Prediction JSON filename inside each Ref folder.",
    )

    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help="Output JSON filename.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    processed = 0
    skipped = 0

    for ref_folder in dataset_root.rglob("Ref"):
        if not ref_folder.is_dir():
            continue

        ok = process_ref_folder(
            ref_folder=ref_folder,
            pred_filename=args.pred_filename,
            output_filename=args.output_filename,
        )

        if ok:
            processed += 1
        else:
            skipped += 1

    print("\nFinished.", flush=True)
    print(f"Processed Ref folders: {processed}", flush=True)
    print(f"Skipped Ref folders: {skipped}", flush=True)


if __name__ == "__main__":
    main()