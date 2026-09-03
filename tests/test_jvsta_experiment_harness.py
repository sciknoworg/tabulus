from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.jvsta.common import (
    flatten_run_metadata,
    numeric_summary,
    parse_gnu_time_verbose,
    select_allocated_gpus,
    summarize_batch_summaries,
    summarize_classification_manifests,
    summarize_output_complexity,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_numeric_summary_uses_interpolated_percentiles() -> None:
    result = numeric_summary([1, 2, 3, 4])
    assert result["median"] == 2.5
    assert result["q25"] == pytest.approx(1.75)
    assert result["q75"] == pytest.approx(3.25)
    assert result["iqr"] == pytest.approx(1.5)
    assert result["p95"] == pytest.approx(3.85)


def test_batch_summary_aggregation_and_failure_taxonomy(tmp_path: Path) -> None:
    summary = tmp_path / "paper" / "adapter" / "batch_summary.json"
    _write_json(
        summary,
        {
            "crop_root": "/crops/paper",
            "output_dir": str(summary.parent),
            "tables_requested": 5,
            "tables_ok": 3,
            "tables_empty": 1,
            "tables_error": 1,
            "prediction_csvs": 1,
            "elapsed_seconds": 30.0,
            "items": [
                {"status": "ok", "elapsed_seconds": 2.0, "parsed_tables": 1, "prediction_csv": "a.csv", "error": None},
                {"status": "ok", "elapsed_seconds": 3.0, "parsed_tables": 0, "prediction_csv": None, "error": None},
                {"status": "ok", "elapsed_seconds": 4.0, "parsed_tables": 2, "prediction_csv": None, "error": None},
                {"status": "empty", "elapsed_seconds": 10.0, "parsed_tables": 0, "prediction_csv": None, "error": "Reached the max_new_tokens generation ceiling."},
                {"status": "error", "elapsed_seconds": 11.0, "parsed_tables": 0, "prediction_csv": None, "error": "CUDA error"},
            ],
        },
    )

    result = summarize_batch_summaries([summary])
    assert result["tables_requested"] == 5
    assert result["prediction_csvs"] == 1
    assert result["structured_reconstruction_yield"] == pytest.approx(0.2)
    assert result["parsing"] == {"zero": 3, "one": 1, "multiple": 1}
    assert result["failure_taxonomy"] == {
        "generation_ceiling": 1,
        "multiple_table_ambiguity": 1,
        "no_parseable_table": 1,
        "runtime_error": 1,
    }
    assert result["tables_per_minute"] == pytest.approx(10.0)
    assert result["per_table_seconds"]["median"] == 4.0


def test_output_complexity_uses_successful_single_table_results(tmp_path: Path) -> None:
    reconstruction_dir = tmp_path / "paper" / "adapter"
    parsed = reconstruction_dir / "parsed" / "table.json"
    _write_json(
        parsed,
        {
            "tables": [
                {"n_rows": 3, "n_cols": 4, "rows": [["x"]], "source": "html"}
            ]
        },
    )
    summary = reconstruction_dir / "batch_summary.json"
    _write_json(
        summary,
        {
            "items": [
                {
                    "status": "ok",
                    "parsed_tables": 1,
                    "parsed_result": "parsed/table.json",
                },
                {
                    "status": "empty",
                    "parsed_tables": 0,
                    "parsed_result": "parsed/missing.json",
                },
            ]
        },
    )

    result = summarize_output_complexity([summary])
    assert result["tables"] == 1
    assert result["rows"]["median"] == 3.0
    assert result["columns"]["median"] == 4.0
    assert result["cells"]["median"] == 12.0


def test_classification_summary_separates_successful_and_unavailable(tmp_path: Path) -> None:
    manifest = tmp_path / "reference_table_classification.json"
    _write_json(
        manifest,
        {
            "tables": [
                {"source_status": "ok", "parsed_tables": 1, "is_reference_table": True, "classification_source": "heuristic"},
                {"source_status": "ok", "parsed_tables": 1, "is_reference_table": False, "classification_source": "heuristic"},
                {"source_status": "empty", "parsed_tables": 0, "is_reference_table": True, "classification_source": "continued_table"},
                {"source_status": "empty", "parsed_tables": 0, "is_reference_table": False, "classification_source": "heuristic"},
            ]
        },
    )

    result = summarize_classification_manifests([manifest])
    assert result["tables_considered"] == 4
    assert result["reference_decisions_all_physical_tables"] == 2
    assert result["successful_reference_like"] == 1
    assert result["successful_non_reference_like"] == 1
    assert result["inherited_continuation_decisions_all"] == 1
    assert result["unavailable_for_independent_classification"] == 2


def test_gpu_selection_prefers_cuda_visible_devices() -> None:
    inventory = [
        {"index": "0", "uuid": "GPU-aaa", "name": "A", "memory_total_mib": 100, "memory_used_mib": 1},
        {"index": "1", "uuid": "GPU-bbb", "name": "B", "memory_total_mib": 200, "memory_used_mib": 2},
    ]
    selected = select_allocated_gpus(inventory, cuda_visible_devices="1")
    assert [gpu["uuid"] for gpu in selected] == ["GPU-bbb"]


def test_parse_gnu_time_verbose(tmp_path: Path) -> None:
    path = tmp_path / "time.txt"
    path.write_text(
        "\tMaximum resident set size (kbytes): 1048576\n",
        encoding="utf-8",
    )
    result = parse_gnu_time_verbose(path)
    assert result["max_rss_kib"] == 1048576
    assert result["max_rss_mib"] == pytest.approx(1024.0)


def test_flatten_run_metadata_contains_paper_metrics() -> None:
    metadata = {
        "run_id": "r1",
        "corpus": "ald",
        "adapter": "internvl3-5-8b",
        "status": "success",
        "git": {"commit": "abc", "tracked_worktree_dirty": False},
        "host": {"hostname": "node"},
        "slurm": {"slurm_job_partition": "p_48G", "slurm_job_id": "7", "cuda_visible_devices": "0"},
        "environment": {"conda_default_env": "env"},
        "resources": {
            "allocated_gpus": [{"name": "L40S", "memory_total_mib": 46000}],
            "gpu": {"peak_gpu_memory_mib": 16000},
            "max_host_rss_mib": 8000,
        },
        "timing": {"reconstruction_wall_seconds": 3600},
        "reconstruction": {
            "batch_elapsed_seconds": 3500,
            "tables_requested": 83,
            "tables_ok": 80,
            "tables_empty": 3,
            "tables_error": 0,
            "prediction_csvs": 80,
            "structured_reconstruction_yield": 80 / 83,
            "tables_per_minute": 1.4,
            "parsing": {"zero": 3, "one": 80, "multiple": 0},
            "per_table_seconds": {"median": 40, "iqr": 10, "p95": 80},
            "failure_taxonomy": {"generation_ceiling": 3},
        },
        "downstream": {
            "successful_reference_like": 67,
            "successful_non_reference_like": 13,
            "inherited_continuation_decisions_all": 1,
            "unavailable_for_independent_classification": 3,
        },
        "output_complexity": {
            "rows": {"median": 10},
            "columns": {"median": 5},
            "cells": {"median": 50},
        },
        "model_provenance": {"adapter_version": "4.55.0", "model_version": "model@rev"},
        "paths": {"run_dir": "/runs/r1"},
    }
    row = flatten_run_metadata(metadata)
    assert row["partition"] == "p_48G"
    assert row["gpu_models"] == "L40S"
    assert row["gpu_hours"] == pytest.approx(1.0)
    assert row["failure_generation_ceiling"] == 3
    assert row["successful_reference_like"] == 67
    assert row["median_cells"] == 50
