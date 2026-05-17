from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List


GOLD_FILENAME = "gold_standard_references.json"
DEFAULT_PRED_FILENAME = "grobid_prediction.json"
DEFAULT_OUTPUT_FILENAME = "grobid_fuzzy_f1_score_result.json"
DEFAULT_SIMILARITY_THRESHOLD = 0.55


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_ref(ref: str) -> str:
    ref = ref.lower()
    ref = re.sub(r"\[.*?\]", " ", ref)
    ref = re.sub(r"https?://\S+", " ", ref)
    ref = re.sub(r"\bdoi\b", " ", ref)
    ref = re.sub(r"[^\w\s]", " ", ref)
    ref = re.sub(r"\s+", " ", ref)
    return ref.strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def extract_refs(data: Any) -> List[str]:
    if not isinstance(data, list):
        return []

    refs: List[str] = []

    for item in data:
        if isinstance(item, dict) and isinstance(item.get("ref"), str):
            refs.append(normalize_ref(item["ref"]))

    return refs


def calculate_fuzzy_f1(
    gold_data: Any,
    pred_data: Any,
    similarity_threshold: float,
) -> Dict[str, Any]:
    gold_refs = extract_refs(gold_data)
    pred_refs = extract_refs(pred_data)

    matched_pred_indexes = set()
    matches = []

    for gold_index, gold_ref in enumerate(gold_refs):
        best_score = 0.0
        best_pred_index = None

        for pred_index, pred_ref in enumerate(pred_refs):
            if pred_index in matched_pred_indexes:
                continue

            score = similarity(gold_ref, pred_ref)

            if score > best_score:
                best_score = score
                best_pred_index = pred_index

        if best_pred_index is not None and best_score >= similarity_threshold:
            matched_pred_indexes.add(best_pred_index)
            matches.append(
                {
                    "gold_nr": str(gold_index + 1),
                    "pred_nr": str(best_pred_index + 1),
                    "similarity": round(best_score, 4),
                }
            )

    tp = len(matches)
    fp = len(pred_refs) - tp
    fn = len(gold_refs) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "gold_total": len(gold_refs),
        "prediction_total": len(pred_refs),
        "similarity_threshold": similarity_threshold,
        "matches": matches,
    }


def process_ref_folder(
    ref_folder: Path,
    pred_filename: str,
    output_filename: str,
    similarity_threshold: float,
) -> Dict[str, Any] | None:
    gold_path = ref_folder / GOLD_FILENAME
    pred_path = ref_folder / pred_filename
    output_path = ref_folder / output_filename

    if not gold_path.exists() or not pred_path.exists():
        return None

    gold_data = load_json(gold_path)
    pred_data = load_json(pred_path)

    result = calculate_fuzzy_f1(
        gold_data=gold_data,
        pred_data=pred_data,
        similarity_threshold=similarity_threshold,
    )

    result["input_files"] = {
        "gold": str(gold_path),
        "prediction": str(pred_path),
    }

    write_json(output_path, result)

    print(f"Saved: {output_path}", flush=True)
    print(f"F1: {result['f1_score']:.3f}", flush=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GROBID reference extraction with fuzzy F1.")
    parser.add_argument("--dataset-root", required=True, help="Path to dataset root.")
    parser.add_argument("--pred-filename", default=DEFAULT_PRED_FILENAME)
    parser.add_argument("--output-filename", default=DEFAULT_OUTPUT_FILENAME)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    processed = 0
    skipped = 0

    for ref_folder in dataset_root.rglob("Ref"):
        if not ref_folder.is_dir():
            continue

        result = process_ref_folder(
            ref_folder=ref_folder,
            pred_filename=args.pred_filename,
            output_filename=args.output_filename,
            similarity_threshold=args.threshold,
        )

        if result is None:
            skipped += 1
        else:
            processed += 1

    print("\nFinished.", flush=True)
    print(f"Processed Ref folders: {processed}", flush=True)
    print(f"Skipped Ref folders: {skipped}", flush=True)


if __name__ == "__main__":
    main()