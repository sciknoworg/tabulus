from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_nr(value: Any) -> int:
    return int(str(value).strip())


def flatten_refs(refs: List[Any]) -> List[int]:
    flat = []

    for item in refs:
        if isinstance(item, list):
            for sub_item in item:
                flat.append(normalize_nr(sub_item))
        else:
            flat.append(normalize_nr(item))

    return flat


def evaluate(
    extracted_path: Path,
    gold_path: Path,
    table_refs_path: Path,
    output_path: Path,
) -> None:
    extracted_data = load_json(extracted_path)
    gold_data = load_json(gold_path)
    table_ref_column = load_json(table_refs_path)

    table_ref_flat = flatten_refs(table_ref_column)

    gold_all_set = {
        normalize_nr(item["nr"])
        for item in gold_data
        if isinstance(item, dict) and item.get("ref")
    }

    gold_target_set = set(table_ref_flat) & gold_all_set

    predicted_all = {
        normalize_nr(item["nr"])
        for item in extracted_data
        if isinstance(item, dict) and "nr" in item
    }

    predicted_valid = {
        normalize_nr(item["nr"])
        for item in extracted_data
        if isinstance(item, dict) and item.get("ref")
    }

    tp = len(predicted_valid & gold_target_set)
    fp = len(predicted_valid - gold_target_set)
    fn = len(gold_target_set - predicted_valid)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        "input_files": {
            "extracted": str(extracted_path),
            "gold": str(gold_path),
            "table_refs": str(table_refs_path),
        },
        "table_refs_total_raw": len(table_ref_column),
        "table_refs_total_flat": len(table_ref_flat),
        "gold_target_refs": len(gold_target_set),
        "total_predictions": len(predicted_all),
        "valid_predictions": len(predicted_valid),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "target_refs_flat": sorted(table_ref_flat),
        "missing_refs_fn": sorted(gold_target_set - predicted_valid),
        "wrong_refs_fp": sorted(predicted_valid - gold_target_set),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, metrics)

    print(f"Saved evaluation to: {output_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether extracted references cover the references used in a table."
    )

    parser.add_argument("--extracted", required=True, help="Path to extracted references JSON.")
    parser.add_argument("--gold", required=True, help="Path to gold_standard_references.json.")
    parser.add_argument("--table-refs", required=True, help="JSON file containing table reference numbers.")
    parser.add_argument("--out", required=True, help="Output JSON path.")

    args = parser.parse_args()

    evaluate(
        extracted_path=Path(args.extracted).resolve(),
        gold_path=Path(args.gold).resolve(),
        table_refs_path=Path(args.table_refs).resolve(),
        output_path=Path(args.out).resolve(),
    )


if __name__ == "__main__":
    main()