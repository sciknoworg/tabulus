"""Score saved TabulusBench Step 2 reconstructions with Exact Mapping Match."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from tabulus.evaluation import evaluate_table_reconstruction
from tabulus.evaluation.rms import csv_to_rms_text
from tabulus.table_ocr.parsing import parse_native_markdown


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None and path.is_file() else None


def native_prediction(
    row: dict[str, str], root: Path, native_root: Path, saved: dict, out: Path,
) -> tuple[Path | None, str]:
    """Reuse archived alignment, or a unique identity/page match for added papers."""
    key = (row["paper_id"], int(row["table_id"]))
    archived = saved.get(key)
    if archived is not None:
        value = archived["prediction_csv"]
        if archived["status"] != "ok" or not value:
            return None, archived["status"]
        return native_root / value, "saved_native_csv"

    paper = (root / row["gold_table_path"]).parents[3]
    index = paper / "tabulus_runs/step1/production/artifacts/table-crops/tables_index.json"
    if not index.is_file():
        return None, "missing_native_index"
    legacy = int(row.get("legacy_table_number") or row["table_id"])
    candidates = [
        item for item in load(index)["tables"]
        if item["table_id"] == legacy and int(item["page_nr"]) == int(row["page_nr"])
    ]
    if len(candidates) != 1:
        return None, "unmatched_native_fragment"
    body = candidates[0].get("mineru_table_body")
    if not body:
        return None, "missing_native_body"
    tables = parse_native_markdown(body)
    if len(tables) != 1:
        return None, "unusable_native_body"
    path = out / "native_predictions" / row["paper_id"] / f"table_{int(row['table_id']):03d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(tables[0].rows)
    return path, "saved_step1_body_identity_page"


def score_benchmark(
    root: Path, out: Path, *, run_id: str = "run_01",
    adapters: list[str] | None = None, expected_fragments: int = 610,
    native_root: Path | None = None,
) -> dict:
    root, out = root.expanduser().resolve(), out.expanduser().resolve()
    if out.is_relative_to(root) and not out.is_relative_to(root / "results"):
        raise ValueError("Place evaluation output outside benchmark inputs or under results/.")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Evaluation output directory must be empty: {out}")
    with (root / "metadata/tables.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(row["paper_id"], row["table_key"]) for row in rows]
    if len(rows) != expected_fragments or len(set(keys)) != len(rows):
        raise ValueError(f"Expected {expected_fragments} distinct gold fragments, found {len(rows)}.")
    for row in rows:
        gold = (root / row["gold_table_path"]).resolve()
        if not gold.is_relative_to(root):
            raise ValueError(f"Gold path escapes benchmark root: {gold}")
        csv_to_rms_text(gold)  # Fail on invalid gold before scoring predictions.

    controlled = root / "results/step2/controlled" / run_id
    if adapters is None:
        adapters = sorted(
            item.name for item in controlled.iterdir()
            if item.is_dir() and (item / "rms/per-table.json").is_file()
        )
    if not adapters:
        raise ValueError("No saved controlled adapters were selected.")
    if any(not re.fullmatch(r"[\w-]+", value) for value in [run_id, *adapters]):
        raise ValueError("Invalid run or adapter name.")
    native_root = native_root or root / "results/step1/controlled/evaluation/mineru-native-252-final"
    native_root = native_root.expanduser().resolve()
    native_manifest = native_root / "table_results.csv"
    native = {}
    if native_manifest.is_file():
        with native_manifest.open(encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                key = (item["paper"], int(item["table_id"]))
                if key in native:
                    raise ValueError(f"Duplicate native fragment: {key}")
                native[key] = item

    out.mkdir(parents=True, exist_ok=True)
    methods = [*adapters, *(["mineru-native"] if native_manifest.is_file() else [])]
    summary = {
        "schema_version": 1, "metric": "exact-mapping", "score_scale": "[0,100]",
        "run_id": run_id, "fragments": len(rows), "inference_rerun": False,
        "aggregation": "Unweighted macro mean over all gold fragments. Missing or unusable predictions score zero.",
        "metadata_sha256": digest(root / "metadata/tables.csv"),
        "native_alignment_manifest": str(native_manifest) if native_manifest.is_file() else None,
        "native_alignment_sha256": digest(native_manifest),
        "native_scope": "Saved Step 1 native outputs use their own detected crops.",
        "methods": {},
    }
    for adapter in methods:
        archived = {}
        status_manifest = controlled / adapter / "rms/per-table.json"
        if adapter != "mineru-native" and status_manifest.is_file():
            archived = {
                (item["paper_id"], int(item["table_id"])): item["status"]
                for item in load(status_manifest)
            }
        records = []
        for row in rows:
            gold = (root / row["gold_table_path"]).resolve()
            paper = gold.parents[3]
            tid = int(row["table_id"])
            if adapter == "mineru-native":
                prediction, evidence = native_prediction(row, root, native_root, native, out)
                prior_status = "ok"
            else:
                prediction = paper / f"tabulus_runs/step2/controlled/{run_id}/{adapter}/reconstruction/predictions/table_{tid:03d}.csv"
                evidence = "saved_controlled_csv"
                prior_status = archived.get((row["paper_id"], tid), "ok")
            values = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            status, error = "missing", None
            if prior_status != "ok":
                status = "unusable"
                error = f"Archived prediction status is {prior_status}."
            elif prediction is not None and prediction.is_file():
                try:
                    if csv_to_rms_text(prediction).strip():
                        score = evaluate_table_reconstruction(gold, prediction, metric="exact-mapping")
                        values = {field: getattr(score, field) for field in values}
                        status = "scored"
                    else:
                        status = "unusable"
                        error = "Empty prediction CSV."
                except (csv.Error, UnicodeError, ValueError) as exc:
                    status, error = "unusable", str(exc)
            records.append({
                "paper_id": row["paper_id"], "table_key": row["table_key"],
                "table_id": tid, "gold_csv": str(gold), "gold_sha256": digest(gold),
                "prediction_csv": str(prediction) if prediction is not None else None,
                "prediction_sha256": digest(prediction), "status": status,
                "source": evidence, "error": error, **values,
            })
        report = {
            "schema_version": 1, "metric": "exact-mapping", "score_scale": "[0,100]",
            "adapter": adapter, "fragments": len(rows),
            "coverage": dict(Counter(item["status"] for item in records)),
            "macro": {field: mean(item[field] for item in records) for field in values},
            "saved_status_manifest_sha256": digest(status_manifest),
            "tables": records,
        }
        (out / f"{adapter}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary["methods"][adapter] = {key: value for key, value in report.items() if key != "tables"}
        print(adapter, json.dumps(report["macro"]), json.dumps(report["coverage"]), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["adapter", "fragments", "scored", "precision", "recall", "f1"])
        for adapter, report in summary["methods"].items():
            writer.writerow([adapter, len(rows), report["coverage"].get("scored", 0), *report["macro"].values()])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="TabulusBench root.")
    parser.add_argument("--out", required=True, type=Path, help="New evaluation output directory.")
    parser.add_argument("--run-id", default="run_01")
    parser.add_argument("--adapters", nargs="+", help="Default to all archived controlled adapters.")
    parser.add_argument("--expected-fragments", type=int, default=610)
    parser.add_argument("--native-results", type=Path, help="Saved MinerU-native evaluation directory.")
    args = parser.parse_args()
    score_benchmark(
        args.root, args.out, run_id=args.run_id, adapters=args.adapters,
        expected_fragments=args.expected_fragments, native_root=args.native_results,
    )


if __name__ == "__main__":
    main()
