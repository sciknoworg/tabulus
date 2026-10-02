from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
import csv
import json

from tabulus.evaluation import evaluate_bibliography


ROOT = Path.home() / "tabulusbench"

OUT = (
    ROOT
    / "results"
    / "step4"
    / "production"
)

PRODUCTION_RUN = "run_01"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def coverage_prf(
    gold_used: int,
    pred_used: int,
    gold_n: int,
    pred_n: int,
) -> tuple[float, float, float]:
    precision = (
        pred_used / pred_n
        if pred_n
        else 0.0
    )

    recall = (
        gold_used / gold_n
        if gold_n
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def paper_number(path: Path) -> int:
    paper = path.parent.parent.name

    if not (
        paper.startswith("P")
        and paper[1:].isdigit()
    ):
        raise ValueError(
            f"Unexpected paper directory: {paper}"
        )

    return int(paper[1:])


def read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


# ----------------------------------------------------------------------
# Preflight: Step 4 gold evaluation population
# ----------------------------------------------------------------------

gold_files = sorted(
    ROOT.rglob("bibliography/gold.json"),
    key=paper_number,
)

if len(gold_files) != 52:
    raise RuntimeError(
        "Expected exactly 52 Step 4 gold bibliographies; "
        f"found {len(gold_files)}."
    )

gold_papers = {
    path.parent.parent.name
    for path in gold_files
}

if len(gold_papers) != 52:
    raise RuntimeError(
        "Step 4 gold discovery contains duplicate paper IDs."
    )


# ----------------------------------------------------------------------
# Preflight: full 252-paper production artifacts
# ----------------------------------------------------------------------

metadata_files = sorted(
    ROOT.rglob(
        "P*/tabulus_runs/step4/"
        "production/run_01/metadata.json"
    ),
    key=lambda p: int(
        p.parents[4].name[1:]
    ),
)

if len(metadata_files) != 252:
    raise RuntimeError(
        "Expected exactly 252 Step 4 production metadata files; "
        f"found {len(metadata_files)}."
    )

expected_ids = {
    f"P{i}"
    for i in range(1, 253)
}

actual_ids = {
    path.parents[4].name
    for path in metadata_files
}

if actual_ids != expected_ids:
    missing = sorted(
        expected_ids - actual_ids,
        key=lambda x: int(x[1:]),
    )

    extra = sorted(
        actual_ids - expected_ids,
        key=lambda x: int(x[1:]),
    )

    raise RuntimeError(
        "Step 4 production population is not exactly P1..P252. "
        f"Missing={missing}, extra={extra}"
    )


# ----------------------------------------------------------------------
# Validate every production artifact before calculating anything
# ----------------------------------------------------------------------

production_records = []

for metadata_path in metadata_files:
    paper_dir = metadata_path.parents[4]
    paper = paper_dir.name

    rel = paper_dir.relative_to(ROOT)

    if len(rel.parts) != 3:
        raise RuntimeError(
            f"Unexpected canonical paper path: {paper_dir}"
        )

    domain, subdomain, _ = rel.parts

    prediction_path = (
        paper_dir
        / "tabulus_runs"
        / "step4"
        / "production"
        / PRODUCTION_RUN
        / "references"
        / "bibliography.json"
    )

    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"{paper}: production bibliography missing: "
            f"{prediction_path}"
        )

    metadata = read_json(metadata_path)
    prediction = read_json(prediction_path)

    entries = prediction.get("entries")

    if not isinstance(entries, list):
        raise RuntimeError(
            f"{paper}: invalid production entries list."
        )

    declared = prediction.get(
        "bibliography_count"
    )

    if declared != len(entries):
        raise RuntimeError(
            f"{paper}: bibliography_count={declared} "
            f"but len(entries)={len(entries)}."
        )

    if metadata.get("status") != "complete":
        raise RuntimeError(
            f"{paper}: metadata status is "
            f"{metadata.get('status')!r}, not 'complete'."
        )

    metadata_count = metadata.get(
        "bibliography_count"
    )

    if metadata_count != len(entries):
        raise RuntimeError(
            f"{paper}: metadata bibliography_count="
            f"{metadata_count}, artifact has {len(entries)}."
        )

    production_records.append({
        "paper": paper,
        "domain": domain,
        "subdomain": subdomain,
        "paper_dir": paper_dir,
        "metadata_path": metadata_path,
        "prediction_path": prediction_path,
        "metadata": metadata,
        "predicted_entries": len(entries),
    })


# ----------------------------------------------------------------------
# Evaluate the 52 gold papers using the validated library implementation
# ----------------------------------------------------------------------

paper_rows = []
alignment_rows = []

availability = Counter()

print()
print("=" * 88)
print("STEP 4 PRODUCTION EVALUATION")
print("=" * 88)
print(f"Production papers:       {len(production_records)}")
print(f"Gold evaluation papers:  {len(gold_files)}")
print(f"Output:                  {OUT}")
print()

for number, gold_path in enumerate(
    gold_files,
    start=1,
):
    paper_dir = gold_path.parent.parent
    rel = paper_dir.relative_to(ROOT)

    domain = rel.parts[0]
    subdomain = rel.parts[1]
    paper = rel.parts[2]

    prediction_path = (
        paper_dir
        / "tabulus_runs"
        / "step4"
        / "production"
        / PRODUCTION_RUN
        / "references"
        / "bibliography.json"
    )

    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"{paper}: production prediction missing: "
            f"{prediction_path}"
        )

    result = evaluate_bibliography(
        gold_path,
        prediction_path,
    )

    for key, value in (
        result.predicted_component_availability.items()
    ):
        availability[key] += value

    one_to_one = [
        operation
        for operation in result.operations
        if operation["kind"] == "one_to_one"
    ]

    splits = [
        operation
        for operation in result.operations
        if operation["kind"] == "split"
    ]

    merges = [
        operation
        for operation in result.operations
        if operation["kind"] == "merge"
    ]

    row = {
        "domain": domain,
        "subdomain": subdomain,
        "paper": paper,
        "gold_entries": result.gold_entries,
        "predicted_entries": result.predicted_entries,
        "content_recovered_gold": (
            result.content_recovered_gold
        ),
        "content_supported_predictions": (
            result.content_supported_predictions
        ),
        "content_precision": result.content_precision,
        "content_recall": result.content_recall,
        "content_f1": result.content_f1,
        "strict_entry_true_positives": (
            result.strict_entry_true_positives
        ),
        "strict_entry_false_positives": (
            result.strict_entry_false_positives
        ),
        "strict_entry_false_negatives": (
            result.strict_entry_false_negatives
        ),
        "strict_entry_precision": (
            result.strict_entry_precision
        ),
        "strict_entry_recall": (
            result.strict_entry_recall
        ),
        "strict_entry_f1": (
            result.strict_entry_f1
        ),
        "one_to_one_gold": result.one_to_one_gold,
        "split_gold_groups": (
            result.split_gold_groups
        ),
        "split_pred_entries": (
            result.split_pred_entries
        ),
        "merge_groups": result.merge_groups,
        "merged_gold_entries": (
            result.merged_gold_entries
        ),
        "unmatched_gold": result.unmatched_gold,
        "unmatched_pred": result.unmatched_pred,
        "position_preserved_gold": (
            result.position_preserved_gold
        ),
        "position_preservation_rate": (
            result.position_preservation_rate
        ),
        "mean_abs_position_error_1to1": (
            result.mean_abs_position_error_1to1
        ),
        "median_abs_position_error_1to1": (
            result.median_abs_position_error_1to1
        ),
        "mean_alignment_score": (
            result.mean_alignment_score
        ),
    }

    paper_rows.append(row)

    for operation in result.operations:
        alignment_rows.append({
            "domain": domain,
            "subdomain": subdomain,
            "paper": paper,
            "kind": operation.get("kind"),
            "gold_start": operation.get(
                "gold_start"
            ),
            "gold_count": operation.get(
                "gold_count"
            ),
            "pred_start": operation.get(
                "pred_start"
            ),
            "pred_count": operation.get(
                "pred_count"
            ),
            "score": operation.get("score"),
            "component_score": operation.get(
                "component_score"
            ),
            "block_precision": operation.get(
                "block_precision"
            ),
            "block_recall": operation.get(
                "block_recall"
            ),
            "block_f1": operation.get(
                "block_f1"
            ),
            "positive_families": operation.get(
                "positive_families"
            ),
            "doi_score": operation.get(
                "doi_score"
            ),
            "authors_score": operation.get(
                "authors_score"
            ),
            "year_score": operation.get(
                "year_score"
            ),
            "title_score": operation.get(
                "title_score"
            ),
            "venue_score": operation.get(
                "venue_score"
            ),
            "locator_score": operation.get(
                "locator_score"
            ),
            "position_preserved": (
                operation["kind"] == "one_to_one"
                and operation["gold_start"]
                == operation["pred_start"]
            ),
        })

    print(
        f"[{number:02d}/52] {paper:<4} "
        f"gold={result.gold_entries:<4} "
        f"pred={result.predicted_entries:<4} "
        f"Content P={100 * result.content_precision:7.3f}% "
        f"R={100 * result.content_recall:7.3f}% "
        f"F1={100 * result.content_f1:7.3f}% "
        f"| Strict F1={100 * result.strict_entry_f1:7.3f}%"
    )


# ----------------------------------------------------------------------
# Domain aggregation
# ----------------------------------------------------------------------

by_domain = defaultdict(list)

for row in paper_rows:
    by_domain[row["domain"]].append(row)

domain_rows = []

for domain, rows in sorted(
    by_domain.items()
):
    gold_n = sum(
        row["gold_entries"]
        for row in rows
    )

    pred_n = sum(
        row["predicted_entries"]
        for row in rows
    )

    gold_used = sum(
        row["content_recovered_gold"]
        for row in rows
    )

    pred_used = sum(
        row["content_supported_predictions"]
        for row in rows
    )

    (
        micro_p,
        micro_r,
        micro_f1,
    ) = coverage_prf(
        gold_used,
        pred_used,
        gold_n,
        pred_n,
    )

    strict_entry_true_positives = sum(
        row["strict_entry_true_positives"]
        for row in rows
    )

    strict_entry_false_positives = (
        pred_n
        - strict_entry_true_positives
    )

    strict_entry_false_negatives = (
        gold_n
        - strict_entry_true_positives
    )

    (
        micro_strict_entry_precision,
        micro_strict_entry_recall,
        micro_strict_entry_f1,
    ) = coverage_prf(
        strict_entry_true_positives,
        strict_entry_true_positives,
        gold_n,
        pred_n,
    )

    position_preserved = sum(
        row["position_preserved_gold"]
        for row in rows
    )

    domain_rows.append({
        "domain": domain,
        "papers": len(rows),
        "gold_entries": gold_n,
        "predicted_entries": pred_n,
        "content_recovered_gold": gold_used,
        "micro_content_precision": micro_p,
        "micro_content_recall": micro_r,
        "micro_content_f1": micro_f1,
        "strict_entry_true_positives": (
            strict_entry_true_positives
        ),
        "strict_entry_false_positives": (
            strict_entry_false_positives
        ),
        "strict_entry_false_negatives": (
            strict_entry_false_negatives
        ),
        "micro_strict_entry_precision": (
            micro_strict_entry_precision
        ),
        "micro_strict_entry_recall": (
            micro_strict_entry_recall
        ),
        "micro_strict_entry_f1": (
            micro_strict_entry_f1
        ),
        "macro_strict_entry_precision": mean(
            row["strict_entry_precision"]
            for row in rows
        ),
        "macro_strict_entry_recall": mean(
            row["strict_entry_recall"]
            for row in rows
        ),
        "macro_strict_entry_f1": mean(
            row["strict_entry_f1"]
            for row in rows
        ),
        "macro_content_precision": mean(
            row["content_precision"]
            for row in rows
        ),
        "macro_content_recall": mean(
            row["content_recall"]
            for row in rows
        ),
        "macro_content_f1": mean(
            row["content_f1"]
            for row in rows
        ),
        "one_to_one_gold": sum(
            row["one_to_one_gold"]
            for row in rows
        ),
        "split_gold_groups": sum(
            row["split_gold_groups"]
            for row in rows
        ),
        "merge_groups": sum(
            row["merge_groups"]
            for row in rows
        ),
        "unmatched_gold": sum(
            row["unmatched_gold"]
            for row in rows
        ),
        "unmatched_pred": sum(
            row["unmatched_pred"]
            for row in rows
        ),
        "position_preserved_gold": (
            position_preserved
        ),
        "position_preservation_rate": (
            position_preserved / gold_n
            if gold_n
            else 0.0
        ),
    })


# ----------------------------------------------------------------------
# Overall 52-paper evaluation aggregation
# ----------------------------------------------------------------------

gold_total = sum(
    row["gold_entries"]
    for row in paper_rows
)

pred_total = sum(
    row["predicted_entries"]
    for row in paper_rows
)

gold_used_total = sum(
    row["content_recovered_gold"]
    for row in paper_rows
)

pred_used_total = sum(
    row["content_supported_predictions"]
    for row in paper_rows
)

(
    micro_p,
    micro_r,
    micro_f1,
) = coverage_prf(
    gold_used_total,
    pred_used_total,
    gold_total,
    pred_total,
)

strict_entry_true_positives = sum(
    row["strict_entry_true_positives"]
    for row in paper_rows
)

strict_entry_false_positives = (
    pred_total
    - strict_entry_true_positives
)

strict_entry_false_negatives = (
    gold_total
    - strict_entry_true_positives
)

(
    micro_strict_entry_precision,
    micro_strict_entry_recall,
    micro_strict_entry_f1,
) = coverage_prf(
    strict_entry_true_positives,
    strict_entry_true_positives,
    gold_total,
    pred_total,
)

position_preserved_total = sum(
    row["position_preserved_gold"]
    for row in paper_rows
)

summary = {
    "papers": len(paper_rows),
    "gold_entries": gold_total,
    "predicted_entries": pred_total,
    "content_recovered_gold": (
        gold_used_total
    ),
    "content_supported_predictions": (
        pred_used_total
    ),
    "micro_content_precision": micro_p,
    "micro_content_recall": micro_r,
    "micro_content_f1": micro_f1,
    "strict_entry_true_positives": (
        strict_entry_true_positives
    ),
    "strict_entry_false_positives": (
        strict_entry_false_positives
    ),
    "strict_entry_false_negatives": (
        strict_entry_false_negatives
    ),
    "micro_strict_entry_precision": (
        micro_strict_entry_precision
    ),
    "micro_strict_entry_recall": (
        micro_strict_entry_recall
    ),
    "micro_strict_entry_f1": (
        micro_strict_entry_f1
    ),
    "macro_strict_entry_precision": mean(
        row["strict_entry_precision"]
        for row in paper_rows
    ),
    "macro_strict_entry_recall": mean(
        row["strict_entry_recall"]
        for row in paper_rows
    ),
    "macro_strict_entry_f1": mean(
        row["strict_entry_f1"]
        for row in paper_rows
    ),
    "macro_content_precision": mean(
        row["content_precision"]
        for row in paper_rows
    ),
    "macro_content_recall": mean(
        row["content_recall"]
        for row in paper_rows
    ),
    "macro_content_f1": mean(
        row["content_f1"]
        for row in paper_rows
    ),
    "median_paper_content_f1": median(
        row["content_f1"]
        for row in paper_rows
    ),
    "one_to_one_gold": sum(
        row["one_to_one_gold"]
        for row in paper_rows
    ),
    "split_gold_groups": sum(
        row["split_gold_groups"]
        for row in paper_rows
    ),
    "split_pred_entries": sum(
        row["split_pred_entries"]
        for row in paper_rows
    ),
    "merge_groups": sum(
        row["merge_groups"]
        for row in paper_rows
    ),
    "merged_gold_entries": sum(
        row["merged_gold_entries"]
        for row in paper_rows
    ),
    "unmatched_gold": sum(
        row["unmatched_gold"]
        for row in paper_rows
    ),
    "unmatched_pred": sum(
        row["unmatched_pred"]
        for row in paper_rows
    ),
    "position_preserved_gold": (
        position_preserved_total
    ),
    "position_preservation_rate": (
        position_preserved_total
        / gold_total
        if gold_total
        else 0.0
    ),
    "match_threshold": 0.62,
    "block_f1_threshold": 0.55,
    "max_split": 12,
    "max_merge": 6,
    "lookahead": 12,
    "component_weights": {
        "doi": 4.0,
        "authors": 3.0,
        "year": 1.5,
        "title": 2.5,
        "venue": 1.0,
        "locator": 1.5,
    },
    "predicted_component_availability": {
        key: availability[key]
        for key in (
            "entries",
            "authors",
            "year",
            "title",
            "venue",
            "volume",
            "pages",
            "doi",
        )
    },
}


# ----------------------------------------------------------------------
# Full 252-paper production execution calculations
# ----------------------------------------------------------------------

execution_rows = []
runtimes = []
total_extracted = 0
started_times = []
finished_times = []

for record in production_records:
    metadata = record["metadata"]
    paper = record["paper"]
    paper_dir = record["paper_dir"]

    seconds = float(
        metadata["wall_time_seconds"]
    )

    runtimes.append(seconds)

    total_extracted += record[
        "predicted_entries"
    ]

    started = datetime.fromisoformat(
        metadata["started_at"]
    )

    finished = datetime.fromisoformat(
        metadata["finished_at"]
    )

    started_times.append(started)
    finished_times.append(finished)

    gold_path = (
        paper_dir
        / "bibliography"
        / "gold.json"
    )

    if gold_path.is_file():
        gold_entries = len(
            read_json(gold_path)
        )
    else:
        gold_entries = ""

    execution_rows.append({
        "paper": paper,
        "path": str(
            paper_dir.relative_to(ROOT)
        ),
        "gold_entries": gold_entries,
        "extracted_entries": (
            record["predicted_entries"]
        ),
        "status": "success",
        "seconds": seconds,
        "output": str(
            record["prediction_path"]
        ),
    })

batch_span_seconds = (
    max(finished_times)
    - min(started_times)
).total_seconds()

production_summary = {
    "papers": len(production_records),
    "successful": len(production_records),
    "failed": 0,
    "bibliography_entries": total_extracted,
    "batch_wall_time_seconds": (
        batch_span_seconds
    ),
    "batch_wall_time_minutes": (
        batch_span_seconds / 60.0
    ),
    "per_paper_runtime_seconds": {
        "mean": mean(runtimes),
        "median": median(runtimes),
        "min": min(runtimes),
        "max": max(runtimes),
        "sum": sum(runtimes),
    },
}


# ----------------------------------------------------------------------
# Write calculated results only
# ----------------------------------------------------------------------

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

paper_csv = (
    OUT / "step4-paper-results.csv"
)

domain_csv = (
    OUT / "step4-domain-results.csv"
)

alignment_csv = (
    OUT / "step4-alignments.csv"
)

summary_json = (
    OUT / "summary.json"
)

execution_csv = (
    OUT / "step4-execution-summary.csv"
)

production_summary_json = (
    OUT / "production-run-summary.json"
)


with paper_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=paper_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(paper_rows)


with domain_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=domain_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(domain_rows)


alignment_fields = [
    "domain",
    "subdomain",
    "paper",
    "kind",
    "gold_start",
    "gold_count",
    "pred_start",
    "pred_count",
    "score",
    "component_score",
    "block_precision",
    "block_recall",
    "block_f1",
    "positive_families",
    "doi_score",
    "authors_score",
    "year_score",
    "title_score",
    "venue_score",
    "locator_score",
    "position_preserved",
]

with alignment_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=alignment_fields,
    )
    writer.writeheader()
    writer.writerows(alignment_rows)


summary_json.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


with execution_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "paper",
            "path",
            "gold_entries",
            "extracted_entries",
            "status",
            "seconds",
            "output",
        ],
    )
    writer.writeheader()
    writer.writerows(execution_rows)


production_summary_json.write_text(
    json.dumps(
        production_summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

print()
print("=" * 88)
print("STEP 4 PRODUCTION EVALUATION COMPLETE")
print("=" * 88)
print()
print("Full production run:")
print(
    f"  Papers:                "
    f"{production_summary['papers']}"
)
print(
    f"  Successful:            "
    f"{production_summary['successful']}"
)
print(
    f"  Bibliography entries:  "
    f"{production_summary['bibliography_entries']}"
)
print(
    f"  Batch wall time:       "
    f"{production_summary['batch_wall_time_minutes']:.2f} min"
)
print(
    f"  Mean / paper:          "
    f"{production_summary['per_paper_runtime_seconds']['mean']:.2f} s"
)
print(
    f"  Median / paper:        "
    f"{production_summary['per_paper_runtime_seconds']['median']:.2f} s"
)
print()
print("Gold evaluation:")
print(
    f"  Papers:                "
    f"{summary['papers']}"
)
print(
    f"  Gold entries:          "
    f"{summary['gold_entries']}"
)
print(
    f"  Predicted entries:     "
    f"{summary['predicted_entries']}"
)
print(
    f"  Recovered gold:        "
    f"{summary['content_recovered_gold']}"
)
print(
    f"  Supported predictions: "
    f"{summary['content_supported_predictions']}"
)
print(
    f"  Micro precision:       "
    f"{100 * summary['micro_content_precision']:.4f}%"
)
print(
    f"  Micro recall:          "
    f"{100 * summary['micro_content_recall']:.4f}%"
)
print(
    f"  Micro F1:              "
    f"{100 * summary['micro_content_f1']:.4f}%"
)
print(
    f"  Macro F1:              "
    f"{100 * summary['macro_content_f1']:.4f}%"
)
print()
print("Strict bibliography-entry recovery:")
print(
    f"  1:1 true positives:    "
    f"{summary['strict_entry_true_positives']}"
)
print(
    f"  False positives:       "
    f"{summary['strict_entry_false_positives']}"
)
print(
    f"  False negatives:       "
    f"{summary['strict_entry_false_negatives']}"
)
print(
    f"  Micro precision:       "
    f"{100 * summary['micro_strict_entry_precision']:.4f}%"
)
print(
    f"  Micro recall:          "
    f"{100 * summary['micro_strict_entry_recall']:.4f}%"
)
print(
    f"  Micro F1:              "
    f"{100 * summary['micro_strict_entry_f1']:.4f}%"
)
print(
    f"  Macro precision:       "
    f"{100 * summary['macro_strict_entry_precision']:.4f}%"
)
print(
    f"  Macro recall:          "
    f"{100 * summary['macro_strict_entry_recall']:.4f}%"
)
print(
    f"  Macro F1:              "
    f"{100 * summary['macro_strict_entry_f1']:.4f}%"
)

print()
print("Outputs:")
print(f"  {summary_json}")
print(f"  {paper_csv}")
print(f"  {domain_csv}")
print(f"  {alignment_csv}")
print(f"  {execution_csv}")
print(f"  {production_summary_json}")
print("=" * 88)
