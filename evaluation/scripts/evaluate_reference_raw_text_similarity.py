from __future__ import annotations

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List


THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

DEFAULT_OUTPUT_JSON = "reference_eval_results.json"

DEFAULT_REFERENCES_FILE = "references.json"

ONLY_REF_FOLDERS = True


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "ﬂ": "fl",
        "ﬁ": "fi",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.lower()

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)

    return text.strip()


def extract_model_name(filename: str) -> str:
    name = Path(filename).stem

    match = re.match(r"(.+?)_ref_(raw|prediction)_", name)

    if match:
        return match.group(1)

    return name


def best_similarity_in_text(reference: str, raw_text: str) -> float:
    ref_norm = normalize_text(reference)
    raw_norm = normalize_text(raw_text)

    if not ref_norm or not raw_norm:
        return 0.0

    if ref_norm in raw_norm:
        return 1.0

    snippet = ref_norm[: min(len(ref_norm), 180)]

    if len(snippet) < 25:
        return SequenceMatcher(
            None,
            ref_norm,
            raw_norm,
            autojunk=False,
        ).ratio()

    window_size = max(len(snippet) + 40, 140)

    step = 40

    best_score = 0.0

    for i in range(
        0,
        max(1, len(raw_norm) - window_size + 1),
        step,
    ):
        chunk = raw_norm[i : i + window_size]

        score = SequenceMatcher(
            None,
            snippet,
            chunk,
            autojunk=False,
        ).ratio()

        if score > best_score:
            best_score = score

            if best_score >= 0.99:
                break

    return best_score


def load_references(references_file: Path) -> List[Dict[str, str]]:
    data = json.loads(references_file.read_text(encoding="utf-8"))

    references = []

    for item in data:
        nr = str(item.get("nr", "")).strip()
        ref = str(item.get("ref", "")).strip()

        if ref:
            references.append(
                {
                    "nr": nr,
                    "ref": ref,
                }
            )

    return references


def should_process_folder(
    folder: Path,
    references_filename: str,
) -> bool:
    if ONLY_REF_FOLDERS and "ref" not in folder.name.lower():
        return False

    if not (folder / references_filename).exists():
        return False

    raw_files = list(folder.glob("*_ref_raw_*.txt"))
    pred_files = list(folder.glob("*_ref_prediction_*.txt"))

    return len(raw_files) + len(pred_files) > 0


def evaluate_folder(
    folder: Path,
    references_filename: str,
) -> Dict[str, Any]:
    references_file = folder / references_filename

    references = load_references(references_file)

    raw_files = list(folder.glob("*_ref_raw_*.txt"))
    pred_files = list(folder.glob("*_ref_prediction_*.txt"))

    all_files = raw_files + pred_files

    result = {
        "ref_folder": str(folder),
        "num_references": len(references),
        "thresholds_tested": THRESHOLDS,
        "models": {},
    }

    if not references or not all_files:
        return result

    print(f"  References loaded: {len(references)}", flush=True)
    print(f"  Files found: {len(all_files)}", flush=True)

    for file in all_files:
        model_name = extract_model_name(file.name)

        print(f"  Checking model: {model_name} ({file.name})", flush=True)

        try:
            raw_text = file.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception as error:
            print(f"    Error reading file: {error}", flush=True)
            continue

        reference_scores = []

        detailed_refs = []

        for idx, ref_item in enumerate(references, start=1):
            try:
                score = best_similarity_in_text(
                    ref_item["ref"],
                    raw_text,
                )
            except Exception as error:
                print(f"    Error in similarity: {error}", flush=True)
                score = 0.0

            reference_scores.append(score)

            detailed_refs.append(
                {
                    "nr": ref_item["nr"],
                    "score": round(score, 4),
                }
            )

            if idx % 10 == 0 or idx == len(references):
                print(
                    f"    processed {idx}/{len(references)} references",
                    flush=True,
                )

        threshold_stats = {}

        total_refs = len(reference_scores)

        for threshold in THRESHOLDS:
            found_count = sum(
                1 for score in reference_scores
                if score >= threshold
            )

            accuracy = (
                found_count / total_refs
                if total_refs > 0
                else 0.0
            )

            threshold_stats[str(threshold)] = {
                "found": found_count,
                "total": total_refs,
                "accuracy": round(accuracy, 4),
            }

        result["models"][model_name] = {
            "file": file.name,
            "thresholds": threshold_stats,
            "reference_scores": detailed_refs,
        }

    return result


def save_folder_result(
    folder: Path,
    result: Dict[str, Any],
    output_json: str,
) -> None:
    output_path = folder / output_json

    write_json(output_path, result)

    print(f"  Output written to: {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate raw OCR reference extraction similarity."
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to dataset root.",
    )

    parser.add_argument(
        "--references-file",
        default=DEFAULT_REFERENCES_FILE,
        help="Reference JSON filename.",
    )

    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON filename.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Invalid dataset root: {dataset_root}")

    folders = [
        folder
        for folder in dataset_root.rglob("*")
        if folder.is_dir()
        and should_process_folder(
            folder,
            args.references_file,
        )
    ]

    print(f"Folders to process: {len(folders)}", flush=True)

    processed_count = 0

    for folder in folders:
        print(f"\nProcessing: {folder}", flush=True)

        result = evaluate_folder(
            folder,
            args.references_file,
        )

        save_folder_result(
            folder,
            result,
            args.output_json,
        )

        processed_count += 1

    print("\nDone.", flush=True)
    print(f"Processed folders: {processed_count}", flush=True)


if __name__ == "__main__":
    main()