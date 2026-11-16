"""Shared, dataset-independent scoring of prepared physical table grids."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from math import fsum
from statistics import mean
import unicodedata

from tabulus.evaluation.rms import (
    DEFAULT_NUMBER_THRESHOLD, DEFAULT_TEXT_THRESHOLD, RMSScores,
    evaluate_mapping_statistics, mapping_scores, prepare_mappings, rows_to_rms_text,
)

METRICS = ("rms", "exact-mapping", "exact")


@dataclass(frozen=True)
class ExactCellCounts:
    """Nonempty cell counts and whole-grid equality after text normalization."""

    matching_cells: int
    gold_nonempty_cells: int
    predicted_nonempty_cells: int
    exact_grid: bool


def count_exact_cells(gold: Iterable[Sequence[str]], prediction: Iterable[Sequence[str]]) -> ExactCellCounts:
    """Count exact nonempty matches at identical row and column positions."""
    def normalize(rows):
        return [[" ".join(unicodedata.normalize("NFC", cell).split()) for cell in row] for row in rows]
    expected, observed = normalize(gold), normalize(prediction)
    return ExactCellCounts(
        sum(bool(g) and g == p for grow, prow in zip(expected, observed) for g, p in zip(grow, prow)),
        sum(bool(cell) for row in expected for cell in row),
        sum(bool(cell) for row in observed for cell in row),
        expected == observed,
    )


@dataclass(frozen=True)
class PreparedGrid:
    rows: tuple[tuple[str, ...], ...]
    mapping_text: str
    mappings: dict[str, str]
    nonempty_cells: int


def prepare_grid(rows: Iterable[Sequence[str]]) -> PreparedGrid:
    """Preserve source coordinates while preparing the historical RMS representation."""
    rows = tuple(tuple(row) for row in rows)
    text = rows_to_rms_text(rows)
    nonempty = sum(bool(cell.strip()) for row in rows for cell in row)
    return PreparedGrid(rows, text, prepare_mappings(text), nonempty)


def select_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(metrics)
    if not selected or len(set(selected)) != len(selected) or any(m not in METRICS for m in selected):
        raise ValueError(f"Select distinct metrics from {METRICS}.")
    return selected


def score_table_grids(
    gold: PreparedGrid | Iterable[Sequence[str]],
    prediction: PreparedGrid | Iterable[Sequence[str]], *,
    metrics: Sequence[str] = METRICS,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    number_threshold: float = DEFAULT_NUMBER_THRESHOLD,
    status: str = "scored",
) -> dict:
    """Return PRF and sufficient statistics from the same comparisons.

    Direct grid pairs retain single-pair empty conventions. Batch input adapters
    mark missing/empty/unusable predictions explicitly, forcing macro scores to
    zero while keeping the gold denominator for pooled micro recall.
    """
    metrics = select_metrics(metrics)
    if status not in {"scored", "missing", "empty", "unusable"}:
        raise ValueError(f"Unknown prediction status: {status}.")
    if "rms" in metrics and (not 0 <= text_threshold <= 1 or not 0 <= number_threshold <= 1):
        raise ValueError("RMS thresholds must be between zero and one.")
    expected = gold if isinstance(gold, PreparedGrid) else prepare_grid(gold)
    observed = prediction if isinstance(prediction, PreparedGrid) else prepare_grid(prediction)
    if status != "scored":
        observed = prepare_grid([])
    result = {
        "status": status, "metrics": {}, "statistics": {},
        "population": {
            "gold_nonempty_cells": expected.nonempty_cells,
            "predicted_nonempty_cells": observed.nonempty_cells,
            "gold_mapping_count": len(expected.mappings),
        },
    }
    for metric in metrics:
        if metric == "exact":
            cells = count_exact_cells(expected.rows, observed.rows)
            result["exact_cells"] = asdict(cells)
            if status != "scored":
                result["exact_cells"]["exact_grid"] = False
            scores = RMSScores(
                cells.matching_cells / max(1, cells.predicted_nonempty_cells),
                cells.matching_cells / max(1, cells.gold_nonempty_cells),
                2 * cells.matching_cells / max(1, cells.predicted_nonempty_cells + cells.gold_nonempty_cells),
            )
            stats = {
                "gold_count": cells.gold_nonempty_cells,
                "predicted_count": cells.predicted_nonempty_cells,
                "matched_credit": cells.matching_cells,
                "orientation": "original" if status == "scored" else None,
                "exact_grid": cells.exact_grid and status == "scored",
                "unit": "nonempty_cells",
            }
        else:
            mapping = evaluate_mapping_statistics(
                expected.mappings, observed.mapping_text, metric=metric,
                text_threshold=text_threshold, number_threshold=number_threshold,
            )
            scores = mapping.scores
            stats = {
                "gold_count": mapping.gold_count, "predicted_count": mapping.predicted_count,
                "matched_credit": mapping.matched_credit,
                "orientation": mapping.orientation if status == "scored" else None,
                "unit": "mappings",
            }
        result["statistics"][metric] = stats
        result["metrics"][metric] = {
            key: value * 100 if status == "scored" else 0.0 for key, value in asdict(scores).items()
        }
    return result


def aggregate_grid_results(results: Sequence[dict]) -> dict:
    """Pool sufficient statistics, never F1 values, in deterministic identity order."""
    if not results:
        raise ValueError("Cannot aggregate an empty evaluation population.")
    metrics = tuple(results[0]["metrics"])
    if any(set(row["metrics"]) != set(metrics) for row in results):
        raise ValueError("All fragments must use the same metrics.")
    macro, micro, pooled = {}, {}, {}
    for metric in metrics:
        stats = [row["statistics"][metric] for row in results]
        gold = sum(s["gold_count"] for s in stats)
        pred = sum(s["predicted_count"] for s in stats)
        credit = fsum(s["matched_credit"] for s in stats)
        pooled[metric] = {
            "gold_count": gold, "predicted_count": pred, "matched_credit": credit,
            "unit": stats[0]["unit"],
        }
        if metric == "exact":
            scores = RMSScores(credit / max(1, pred), credit / max(1, gold), 2 * credit / max(1, gold + pred))
        else:
            scores = mapping_scores(credit, gold, pred)
        micro[metric] = {key: value * 100 for key, value in asdict(scores).items()}
        macro[metric] = {
            field: mean(row["metrics"][metric][field] for row in results)
            for field in ("precision", "recall", "f1")
        }
    return {
        "macro": macro, "micro": micro, "pooled_counts": pooled,
        "coverage": dict(Counter(row["status"] for row in results)),
        "population": {
            "fragments": len(results),
            **{field: sum(row["population"][field] for row in results) for field in (
                "gold_nonempty_cells", "predicted_nonempty_cells", "gold_mapping_count",
            )},
        },
    }
