"""Score saved controlled, production or resolved grids through the shared evaluator."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import lru_cache
import io
import json
from pathlib import Path, PurePosixPath
from math import isclose
import re

from tabulus.evaluation import FragmentPair, GridInput, evaluate_pairs, load_grid, prepare_grid
from tabulus.evaluation.batch import atomic_json, atomic_text, content_hash
from tabulus.resolved_export import ENRICHMENT_COLUMNS
from tabulus.table_ocr.parsing import parse_native_markdown


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _resolve(directory, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


def _native_input(row, root, native_root, archived):
    key = row["paper_id"], int(row["table_id"])
    saved = archived.get(key)
    if saved is not None:
        if saved["status"] != "ok" or not saved.get("prediction_csv"):
            return None, saved["status"]
        return GridInput(_resolve(native_root, saved["prediction_csv"])), "saved_native_csv"
    paper = (root / row["gold_table_path"]).parents[3]
    index = paper / "tabulus_runs/step1/production/artifacts/table-crops/tables_index.json"
    if not index.is_file():
        return None, "missing_native_index"
    candidates = [
        t for t in load(index)["tables"]
        if row["table_key"] in {
            t.get("benchmark_table_key"), t.get("table_key"),
            PurePosixPath(t.get("image_name", "").replace("\\", "/")).stem,
        } and int(t["page_nr"]) == int(row["page_nr"])
    ]
    if len(candidates) != 1:
        return None, "unmatched_native_fragment"
    body = candidates[0].get("mineru_table_body")
    tables = parse_native_markdown(body) if body else []
    if len(tables) != 1:
        return None, "unusable_native_body"
    return GridInput(index, inline_grid=tuple(tuple(r) for r in tables[0].rows)), "saved_step1_body_identity_page"


def _population(rows, root, cache):
    cells = mappings = 0
    for row in rows:
        path = (root / row["gold_table_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Gold path escapes benchmark root: {path}")
        if path not in cache:
            cache[path] = prepare_grid(load_grid(GridInput(path)))
        grid = cache[path]
        cells += grid.nonempty_cells
        mappings += len(grid.mappings)
    return {"fragments": len(rows), "gold_nonempty_cells": cells, "gold_mapping_count": mappings}



def _verify_archive(records, baseline, selected_metrics):
    """Historical score availability is independent of prediction availability."""
    without, mismatches, not_scored = [], [], []
    checked = overlap = 0
    for row in records:
        identity = {key: row["provenance"][key] for key in ("paper_id", "table_key")}
        key = identity["paper_id"], identity["table_key"]
        archived = baseline["entries"].get(key)
        if archived is None:
            without.append(identity)
            continue
        overlap += 1
        if "rms" not in selected_metrics or not baseline["applicable"]:
            continue
        checked += 1
        if row["status"] != "scored":
            not_scored.append({**identity, "prediction_status": row["status"]})
        current = row["metrics"]["rms"]
        if any(not isclose(current[field], archived[field], rel_tol=0, abs_tol=1e-8)
               for field in ("precision", "recall", "f1")):
            mismatches.append({
                **identity, "prediction_status": row["status"],
                "recalculated": current, "archived": archived,
            })
    return {
        "metric": "rms", "archive_path": baseline["path"],
        "archive_sha256": baseline["sha256"], "archived_fragments": len(baseline["entries"]),
        "selected_fragments": len(records), "overlap": overlap, "checked": checked,
        "without_baseline": without, "mismatches": mismatches,
        "not_scored_with_baseline": not_scored,
        "comparison_enabled": "rms" in selected_metrics and baseline["applicable"],
        "applicable_to_setting": baseline["applicable"],
        "absolute_tolerance": 1e-8,
    }

def score_benchmark(
    root: Path, out: Path, *, run_id="run_01", adapters=None, expected_fragments=610,
    native_root=None, metrics=("exact-mapping",), workers=1, resume=False,
    setting="controlled", subset="all", fragments=None, alignment_path=None, include_native=True,
    text_threshold=.5, number_threshold=.1,
):
    root, out = root.expanduser().resolve(), out.expanduser().resolve()
    if out.is_relative_to(root) and not out.is_relative_to(root / "results"):
        raise ValueError("Place evaluation output outside benchmark inputs or under results/.")
    with (root / "metadata/tables.csv").open(encoding="utf-8-sig", newline="") as handle:
        population = list(csv.DictReader(handle))
    universe = {(r["paper_id"], r["table_key"]): r for r in population}
    ids = {(r["paper_id"], int(r["table_id"])) for r in population}
    if len(population) != expected_fragments or len(universe) != len(population) or len(ids) != len(population):
        raise ValueError(f"Expected {expected_fragments} distinct gold fragments, found {len(population)}.")
    selected = set(universe)
    selection_hash = None
    if subset == "step7":
        selected = set()
        for paper in { (root / r["gold_table_path"]).parents[3] for r in population }:
            path = paper / "step7_resolved_tables/gold.json"
            if path.is_file():
                selected.update((paper.name, t["table_key"]) for t in load(path)["tables"])
        if expected_fragments == 610 and len(selected) != 61:
            raise ValueError(f"Expected the 61-fragment Step 7 population, found {len(selected)}.")
    if fragments is not None:
        fragments = Path(fragments).expanduser().resolve()
        if fragments.suffix.lower() == ".csv":
            with fragments.open(encoding="utf-8-sig", newline="") as handle:
                chosen = list(csv.DictReader(handle))
        else:
            chosen = load(fragments)["fragments"]
        keys = [(r["paper_id"], r["table_key"]) for r in chosen]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate fragment selection.")
        selected &= set(keys)
        if set(keys) - universe.keys():
            raise ValueError("Unknown explicit fragment selection.")
        selection_hash = content_hash(fragments)
    if not selected or selected - universe.keys():
        raise ValueError("Unknown or empty benchmark subset.")
    rows = [r for r in population if (r["paper_id"], r["table_key"]) in selected]
    cache = {}
    full_population, selected_population = _population(population, root, cache), _population(rows, root, cache)
    controlled = root / "results/step2/controlled" / run_id
    if adapters is None:
        adapters = sorted(p.name for p in controlled.iterdir() if (p / "rms/per-table.json").is_file())
    if not adapters or any(not re.fullmatch(r"[\w-]+", value) for value in [run_id, *adapters]):
        raise ValueError("Select valid saved adapter and run names.")
    if setting not in {"controlled", "production", "step7"}:
        raise ValueError("Unknown evaluation setting.")
    correspondence = {}
    alignment_hash = None
    if alignment_path is not None:
        alignment_path = Path(alignment_path).expanduser().resolve()
        alignment_hash = content_hash(alignment_path)
        payload = load(alignment_path)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported alignment schema.")
        claimed = set()
        for item in payload["tables"]:
            key = (item.get("adapter", "*"), item["paper_id"], item["table_key"])
            if key in correspondence or key[1:] not in universe:
                raise ValueError("Duplicate or unknown fragment correspondence.")
            tid = item.get("prediction_table_id", item.get("table_id"))
            if type(tid) is not int:
                raise ValueError("Correspondence requires a prediction_table_id integer.")
            physical = (key[0], key[1], item.get("prediction_manifest", ""), tid)
            if physical in claimed:
                raise ValueError("One prediction cannot represent multiple gold fragments.")
            claimed.add(physical)
            correspondence[key] = {**item, "prediction_table_id": tid}
    if setting != "controlled" and alignment_path is None:
        raise ValueError("Production and physical Step 7 require explicit correspondence.")

    native_root = Path(native_root or root / "results/step1/controlled/evaluation/mineru-native-252-final").expanduser().resolve()
    native_manifest = native_root / "table_results.csv"
    native = {}
    if native_manifest.is_file():
        with native_manifest.open(encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                key = item["paper"], int(item["table_id"])
                if key in native:
                    raise ValueError(f"Duplicate native fragment: {key}")
                native[key] = item
    methods = list(dict.fromkeys([
        *adapters, *(["mineru-native"] if include_native and setting == "controlled" and native_manifest.is_file() else []),
    ]))
    pairs, manifests, used_predictions = [], {}, set()
    baselines = {}
    file_hash = lru_cache(maxsize=None)(content_hash)
    for adapter in methods:
        status_path = controlled / adapter / "rms/per-table.json"
        archived_records = load(status_path) if setting == "controlled" and status_path.is_file() else []
        entries = {}
        baseline_path = native_manifest if adapter == "mineru-native" else status_path
        if setting == "controlled":
            if adapter == "mineru-native":
                for row in population:
                    key = row["paper_id"], row["table_key"]
                    archived = native.get((row["paper_id"], int(row["table_id"])))
                    if archived is not None and all(f"rms_{field}" in archived for field in ("precision", "recall", "f1")):
                        entries[key] = {field: 100 * float(archived[f"rms_{field}"]) for field in ("precision", "recall", "f1")}
            else:
                for archived in archived_records:
                    key = archived["paper_id"], PurePosixPath(archived["gold_csv"].replace("\\", "/")).parent.name
                    if key in entries:
                        raise ValueError(f"Duplicate archived RMS fragment: {key}")
                    entries[key] = {field: archived[field] for field in ("precision", "recall", "f1")}
        baselines[adapter] = {
            "entries": entries,
            "path": str(baseline_path) if setting == "controlled" and baseline_path.is_file() else None,
            "sha256": file_hash(baseline_path) if setting == "controlled" else None,
            "applicable": setting == "controlled",
            "saved_status_manifest_sha256": file_hash(status_path) if setting == "controlled" else None,
        }
        for row in rows:
            gold = (root / row["gold_table_path"]).resolve()
            paper, tid = gold.parents[3], int(row["table_id"])
            provenance = {
                "paper_id": row["paper_id"], "table_key": row["table_key"], "adapter": adapter,
                "setting": setting, "metadata_sha256": file_hash(root / "metadata/tables.csv"),
                "alignment_sha256": alignment_hash, "selection_sha256": selection_hash,
                "benchmark_preparation_sha256": file_hash(Path(__file__)),
                "saved_status_sha256": file_hash(status_path) if setting == "controlled" else None,
            }
            status, source = "ok", "saved_controlled_csv"
            if adapter == "mineru-native":
                prediction, source = _native_input(row, root, native_root, native)
                provenance["native_alignment_sha256"] = file_hash(native_manifest)
            elif setting == "controlled":
                prediction = GridInput(paper / f"tabulus_runs/step2/controlled/{run_id}/{adapter}/reconstruction/predictions/table_{tid:03d}.csv")
            else:
                item = correspondence.get((adapter, row["paper_id"], row["table_key"]),
                                          correspondence.get(("*", row["paper_id"], row["table_key"])))
                prediction = None
                source = f"explicit_{setting}_correspondence"
                if item is not None:
                    predicted_id = item["prediction_table_id"]
                    provenance["prediction_table_id"] = predicted_id
                    if setting == "production":
                        path = (
                            _resolve(alignment_path.parent, item["prediction_csv"])
                            if item.get("prediction_csv") else
                            paper / f"tabulus_runs/step2/production/{run_id}/{adapter}/reconstruction/predictions/table_{predicted_id:03d}.csv"
                        )
                        prediction = GridInput(path)
                    else:
                        path = _resolve(alignment_path.parent, item["prediction_manifest"])
                        if path not in manifests:
                            manifests[path] = load(path)
                        manifest = manifests[path]
                        columns = tuple(manifest["enrichment_columns"])
                        if not columns or any(c not in ENRICHMENT_COLUMNS for c in columns):
                            raise ValueError("Invalid resolved enrichment column declaration.")
                        entries = {t["table_id"]: t for t in manifest["tables"]}
                        if len(entries) != len(manifest["tables"]):
                            raise ValueError("Duplicate physical prediction IDs.")
                        table = entries.get(predicted_id)
                        if table is not None:
                            prediction = GridInput(
                                _resolve(path.parent, table["resolved_csv"]), columns, table.get("header_row_index"),
                            )
                        provenance["prediction_manifest_sha256"] = file_hash(path)
            if prediction is not None:
                identity = (adapter, prediction.path, tid if prediction.inline_grid is not None else None)
                if identity in used_predictions:
                    raise ValueError("A prediction was assigned to more than one gold fragment.")
                used_predictions.add(identity)
            provenance["source"] = source
            pairs.append(FragmentPair(
                f"{setting}/{adapter}/{row['paper_id']}/{row['table_key']}",
                GridInput(gold), prediction, f"{setting}/{adapter}", status, provenance,
            ))

    completed = {}
    finished_count = 0
    def progress(row):
        nonlocal finished_count
        finished_count += 1
        if finished_count % 100 == 0:
            print(f"Completed {finished_count} new fragments, latest {row['fragment_id']}", flush=True)
    def finish(result):
        completed.update(_finish_reports(
            result, out, metrics, methods, setting, pairs, universe,
            full_population, selected_population, run_id, len(rows), baselines,
        ))
    evaluate_pairs(
        pairs, metrics=metrics, workers=workers, output_dir=out, resume=resume,
        text_threshold=text_threshold, number_threshold=number_threshold,
        on_result=progress, on_complete=finish,
    )
    return completed


def _finish_reports(result, out, metrics, methods, setting, pairs, universe,
                    full_population, selected_population, run_id, row_count, baselines):
    methods_report = {}
    for adapter in methods:
        group = f"{setting}/{adapter}"
        report = result["groups"][group]
        records = [r for r in result["results"] if r["group"] == group]
        coverage = Counter(report["coverage"])
        # Keep the old script's empty-as-unusable coverage field available.
        coverage["unusable"] += coverage.pop("empty", 0)
        coverage = {k: v for k, v in coverage.items() if v}
        macro = report["macro"][metrics[0]] if len(metrics) == 1 else report["macro"]
        methods_report[adapter] = {
            **report, "adapter": adapter, "fragments": len(records),
            "schema_version": 1, "score_scale": "[0,100]",
            "saved_status_manifest_sha256": baselines[adapter]["saved_status_manifest_sha256"],
            "coverage": coverage, "macro": macro, "metric": metrics[0] if len(metrics) == 1 else list(metrics),
            "archive_verification": _verify_archive(records, baselines[adapter], metrics),
        }
        compatible_records = []
        for row in records:
            source = row["inputs"]["prediction"]
            identity = row["provenance"]
            compatible_records.append({
                **row, "paper_id": identity["paper_id"], "table_key": identity["table_key"],
                "table_id": int(universe[(identity["paper_id"], identity["table_key"])]["table_id"]),
                "gold_csv": row["inputs"]["gold"]["path"],
                "gold_sha256": row["inputs"]["gold"].get("sha256"),
                "prediction_csv": source.get("path") if source else None,
                "prediction_sha256": source.get("sha256") if source else None,
                "source": identity["source"],
                **(row["metrics"][metrics[0]] if len(metrics) == 1 else {}),
            })
        atomic_json(out / f"{adapter}.json", {**methods_report[adapter], "tables": compatible_records})
        print(adapter, json.dumps(report["macro"]), json.dumps(report["micro"]), flush=True)
    for pair in pairs:
        if pair.prediction is not None and pair.prediction.inline_grid is not None:
            path = out / "native_predictions" / pair.provenance["paper_id"] / f"table_{int(universe[(pair.provenance['paper_id'], pair.provenance['table_key'])]['table_id']):03d}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = io.StringIO(newline="")
            csv.writer(handle).writerows(pair.prediction.inline_grid)
            atomic_text(path, handle.getvalue())
    summary = {
        **{k: v for k, v in result.items() if k != "results"}, "methods": methods_report,
        "run_id": run_id, "setting": setting, "fragments": row_count, "inference_rerun": False,
        "full_population": full_population, "selected_population": selected_population,
        "population_share": {
            key: selected_population[key] / value if value else None for key, value in full_population.items()
        },
    }
    atomic_json(out / "summary.json", summary)
    handle = io.StringIO(newline="")
    writer = csv.writer(handle)
    writer.writerow(["adapter", "metric", "aggregation", "precision", "recall", "f1"])
    for adapter in methods:
        group = result["groups"][f"{setting}/{adapter}"]
        for aggregation in ("macro", "micro"):
            for metric, scores in group[aggregation].items():
                writer.writerow([adapter, metric, aggregation, *scores.values()])
    atomic_text(out / "summary.csv", handle.getvalue())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", default="run_01")
    parser.add_argument("--adapters", nargs="+")
    parser.add_argument("--expected-fragments", type=int, default=610)
    parser.add_argument("--native-results", type=Path)
    parser.add_argument("--no-native", action="store_true")
    parser.add_argument("--metrics", nargs="+", choices=("rms", "exact-mapping", "exact"), default=["exact-mapping"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--setting", choices=("controlled", "production", "step7"), default="controlled")
    parser.add_argument("--subset", choices=("all", "step7"), default="all")
    parser.add_argument("--fragments", type=Path)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--text-threshold", type=float, default=.5)
    parser.add_argument("--number-threshold", type=float, default=.1)
    args = parser.parse_args()
    score_benchmark(
        args.root, args.out, run_id=args.run_id, adapters=args.adapters,
        expected_fragments=args.expected_fragments, native_root=args.native_results,
        metrics=args.metrics, workers=args.workers, resume=args.resume,
        setting=args.setting, subset=args.subset, fragments=args.fragments, alignment_path=args.alignment,
        include_native=not args.no_native, text_threshold=args.text_threshold, number_threshold=args.number_threshold,
    )


if __name__ == "__main__":
    main()
