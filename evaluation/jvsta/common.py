from __future__ import annotations

import csv
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile for ``fraction`` in [0, 1]."""
    if not values:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    cleaned = [float(value) for value in values]
    if not cleaned:
        return {
            "count": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    q25 = percentile(cleaned, 0.25)
    q75 = percentile(cleaned, 0.75)
    return {
        "count": len(cleaned),
        "median": statistics.median(cleaned),
        "q25": q25,
        "q75": q75,
        "iqr": (q75 - q25) if q25 is not None and q75 is not None else None,
        "p95": percentile(cleaned, 0.95),
        "min": min(cleaned),
        "max": max(cleaned),
    }


def failure_category(item: dict[str, Any]) -> str | None:
    status = str(item.get("status", "unknown"))
    parsed_tables = int(item.get("parsed_tables", 0) or 0)
    prediction_csv = item.get("prediction_csv")
    error = str(item.get("error") or "")
    error_lower = error.lower()

    if status == "error":
        return "runtime_error"

    if "ceiling" in error_lower or "max_new_tokens" in error_lower:
        return "generation_ceiling"

    if status == "empty":
        if parsed_tables == 0:
            return "empty_no_structured_output"
        return "empty_other"

    if status == "ok" and parsed_tables == 0:
        return "no_parseable_table"

    if status == "ok" and parsed_tables > 1:
        return "multiple_table_ambiguity"

    if status == "ok" and prediction_csv is None:
        return "no_prediction_other"

    return None


def summarize_batch_summaries(summary_paths: Iterable[Path]) -> dict[str, Any]:
    totals = {
        "tables_requested": 0,
        "tables_ok": 0,
        "tables_empty": 0,
        "tables_error": 0,
        "prediction_csvs": 0,
    }
    parse_counts = {"zero": 0, "one": 0, "multiple": 0}
    failures: dict[str, int] = {}
    per_table_seconds: list[float] = []
    batch_elapsed_seconds = 0.0
    batches: list[dict[str, Any]] = []

    for path in sorted(Path(p) for p in summary_paths):
        payload = load_json_object(path)
        for key in totals:
            totals[key] += int(payload.get(key, 0) or 0)
        batch_elapsed_seconds += float(payload.get("elapsed_seconds", 0.0) or 0.0)

        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Batch summary has no valid items list: {path}")

        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Batch summary contains a non-object item: {path}")
            elapsed = item.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)):
                per_table_seconds.append(float(elapsed))

            parsed_tables = int(item.get("parsed_tables", 0) or 0)
            if parsed_tables == 0:
                parse_counts["zero"] += 1
            elif parsed_tables == 1:
                parse_counts["one"] += 1
            else:
                parse_counts["multiple"] += 1

            category = failure_category(item)
            if category is not None:
                failures[category] = failures.get(category, 0) + 1

        batches.append(
            {
                "summary_path": str(path),
                "crop_root": payload.get("crop_root"),
                "output_dir": payload.get("output_dir"),
                "tables_requested": int(payload.get("tables_requested", 0) or 0),
                "tables_ok": int(payload.get("tables_ok", 0) or 0),
                "tables_empty": int(payload.get("tables_empty", 0) or 0),
                "tables_error": int(payload.get("tables_error", 0) or 0),
                "prediction_csvs": int(payload.get("prediction_csvs", 0) or 0),
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0) or 0.0),
            }
        )

    requested = totals["tables_requested"]
    predictions = totals["prediction_csvs"]
    throughput = (
        requested / (batch_elapsed_seconds / 60.0)
        if requested and batch_elapsed_seconds > 0
        else None
    )

    return {
        **totals,
        "batch_count": len(batches),
        "batch_elapsed_seconds": batch_elapsed_seconds,
        "structured_reconstruction_yield": (
            predictions / requested if requested else None
        ),
        "tables_per_minute": throughput,
        "parsing": parse_counts,
        "per_table_seconds": numeric_summary(per_table_seconds),
        "failure_taxonomy": dict(sorted(failures.items())),
        "batches": batches,
    }


def _resolve_artifact(reconstruction_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else reconstruction_dir / path


def summarize_output_complexity(summary_paths: Iterable[Path]) -> dict[str, Any]:
    rows_values: list[float] = []
    cols_values: list[float] = []
    cells_values: list[float] = []

    for summary_path in sorted(Path(p) for p in summary_paths):
        reconstruction_dir = summary_path.parent
        summary = load_json_object(summary_path)
        items = summary.get("items")
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "ok" or int(item.get("parsed_tables", 0) or 0) != 1:
                continue
            parsed_path = _resolve_artifact(reconstruction_dir, item.get("parsed_result"))
            if parsed_path is None or not parsed_path.is_file():
                continue
            parsed_payload = load_json_object(parsed_path)
            tables = parsed_payload.get("tables")
            if not isinstance(tables, list) or len(tables) != 1 or not isinstance(tables[0], dict):
                continue
            table = tables[0]
            n_rows = int(table.get("n_rows", 0) or 0)
            n_cols = int(table.get("n_cols", 0) or 0)
            rows_values.append(float(n_rows))
            cols_values.append(float(n_cols))
            cells_values.append(float(n_rows * n_cols))

    return {
        "tables": len(rows_values),
        "rows": numeric_summary(rows_values),
        "columns": numeric_summary(cols_values),
        "cells": numeric_summary(cells_values),
    }


def summarize_classification_manifests(manifest_paths: Iterable[Path]) -> dict[str, Any]:
    total_considered = 0
    reference_decisions_all = 0
    successful_reference_like = 0
    successful_non_reference_like = 0
    inherited_all = 0
    inherited_successful = 0
    unavailable = 0
    manifests = 0

    for path in sorted(Path(p) for p in manifest_paths):
        payload = load_json_object(path)
        tables = payload.get("tables")
        if not isinstance(tables, list):
            raise ValueError(f"Classification manifest has no valid tables list: {path}")
        manifests += 1
        total_considered += len(tables)
        for item in tables:
            if not isinstance(item, dict):
                continue
            is_reference = bool(item.get("is_reference_table"))
            if is_reference:
                reference_decisions_all += 1
            is_successful = (
                item.get("source_status") == "ok"
                and int(item.get("parsed_tables", 0) or 0) == 1
            )
            if is_successful:
                if is_reference:
                    successful_reference_like += 1
                else:
                    successful_non_reference_like += 1
            else:
                unavailable += 1

            if item.get("classification_source") == "continued_table":
                inherited_all += 1
                if is_successful:
                    inherited_successful += 1

    return {
        "manifest_count": manifests,
        "tables_considered": total_considered,
        "reference_decisions_all_physical_tables": reference_decisions_all,
        "non_reference_decisions_all_physical_tables": (
            total_considered - reference_decisions_all
        ),
        "successful_reference_like": successful_reference_like,
        "successful_non_reference_like": successful_non_reference_like,
        "inherited_continuation_decisions_all": inherited_all,
        "inherited_continuation_decisions_successful": inherited_successful,
        "unavailable_for_independent_classification": unavailable,
    }


def collect_model_provenance(summary_paths: Iterable[Path]) -> dict[str, Any]:
    adapter_versions: set[str] = set()
    model_versions: set[str] = set()

    for summary_path in sorted(Path(p) for p in summary_paths):
        reconstruction_dir = summary_path.parent
        summary = load_json_object(summary_path)
        items = summary.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            native_path = _resolve_artifact(reconstruction_dir, item.get("native_result"))
            if native_path is None or not native_path.is_file():
                continue
            payload = load_json_object(native_path)
            adapter_version = payload.get("adapter_version")
            model_version = payload.get("model_version")
            if isinstance(adapter_version, str) and adapter_version:
                adapter_versions.add(adapter_version)
            if isinstance(model_version, str) and model_version:
                model_versions.add(model_version)

    return {
        "adapter_version": next(iter(adapter_versions)) if len(adapter_versions) == 1 else None,
        "adapter_versions": sorted(adapter_versions),
        "model_version": next(iter(model_versions)) if len(model_versions) == 1 else None,
        "model_versions": sorted(model_versions),
    }


def git_metadata(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "branch": branch,
        "tracked_worktree_dirty": bool(tracked_status),
        "tracked_status": tracked_status.splitlines() if tracked_status else [],
    }


def slurm_metadata() -> dict[str, str | None]:
    names = [
        "SLURM_JOB_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NODELIST",
        "SLURMD_NODENAME",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {name.lower(): os.environ.get(name) for name in names}


def _parse_nvidia_smi_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for values in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(values) != 5:
            continue
        index, uuid, name, memory_total, memory_used = [value.strip() for value in values]
        try:
            total = float(memory_total)
            used = float(memory_used)
        except ValueError:
            continue
        rows.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_total_mib": total,
                "memory_used_mib": used,
            }
        )
    return rows


def query_gpu_inventory() -> list[dict[str, Any]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []
    completed = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    return _parse_nvidia_smi_rows(completed.stdout)


def select_allocated_gpus(
    inventory: Sequence[dict[str, Any]],
    *,
    cuda_visible_devices: str | None = None,
    slurm_job_gpus: str | None = None,
) -> list[dict[str, Any]]:
    """Best-effort selection of GPUs allocated to this process/job."""
    tokens_text = cuda_visible_devices or slurm_job_gpus or ""
    tokens = [token.strip() for token in tokens_text.split(",") if token.strip()]
    if not tokens:
        return list(inventory)

    selected: list[dict[str, Any]] = []
    for gpu in inventory:
        index = str(gpu.get("index", ""))
        uuid = str(gpu.get("uuid", ""))
        if any(
            token == index
            or uuid.startswith(token)
            or token.startswith(uuid)
            for token in tokens
        ):
            selected.append(dict(gpu))

    if selected:
        return selected

    # Some Slurm configurations remap allocated devices to local CUDA indices.
    # If exactly one GPU is visible to nvidia-smi, that device is unambiguous.
    if len(inventory) == 1:
        return [dict(inventory[0])]
    return list(inventory)


@dataclass
class GpuSampler:
    output_path: Path
    interval_seconds: float = 1.0
    selected_uuids: set[str] = field(default_factory=set)
    peak_by_uuid_mib: dict[str, float] = field(default_factory=dict)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        inventory = query_gpu_inventory()
        selected = select_allocated_gpus(
            inventory,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
        )
        self.selected_uuids = {
            str(gpu.get("uuid"))
            for gpu in selected
            if gpu.get("uuid")
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp_utc",
                    "index",
                    "uuid",
                    "name",
                    "memory_total_mib",
                    "memory_used_mib",
                ]
            )
        if not inventory:
            return
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            inventory = query_gpu_inventory()
            rows = [
                gpu
                for gpu in inventory
                if not self.selected_uuids or str(gpu.get("uuid")) in self.selected_uuids
            ]
            timestamp = utc_now_iso()
            with self.output_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for gpu in rows:
                    uuid = str(gpu.get("uuid", ""))
                    used = float(gpu.get("memory_used_mib", 0.0) or 0.0)
                    self.peak_by_uuid_mib[uuid] = max(
                        used,
                        self.peak_by_uuid_mib.get(uuid, 0.0),
                    )
                    writer.writerow(
                        [
                            timestamp,
                            gpu.get("index"),
                            uuid,
                            gpu.get("name"),
                            gpu.get("memory_total_mib"),
                            used,
                        ]
                    )
            self._stop.wait(max(self.interval_seconds, 0.1))

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds * 2, 2.0))
        peak = max(self.peak_by_uuid_mib.values(), default=None)
        return {
            "peak_gpu_memory_mib": peak,
            "peak_gpu_memory_by_uuid_mib": dict(sorted(self.peak_by_uuid_mib.items())),
            "measurement": "nvidia-smi device-wide memory.used sampled during the run",
            "sample_interval_seconds": self.interval_seconds,
        }


def parse_gnu_time_verbose(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {}
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        value = value.strip()
        if key == "Maximum resident set size (kbytes)":
            try:
                result["max_rss_kib"] = int(value)
                result["max_rss_mib"] = int(value) / 1024.0
            except ValueError:
                pass
    return result


def host_identity() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
    }


def flatten_run_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    reconstruction = metadata.get("reconstruction") or {}
    runtime_stats = reconstruction.get("per_table_seconds") or {}
    parsing = reconstruction.get("parsing") or {}
    failures = reconstruction.get("failure_taxonomy") or {}
    downstream = metadata.get("downstream") or {}
    complexity = metadata.get("output_complexity") or {}
    resources = metadata.get("resources") or {}
    timing = metadata.get("timing") or {}
    git = metadata.get("git") or {}
    slurm = metadata.get("slurm") or {}
    provenance = metadata.get("model_provenance") or {}
    gpu_inventory = resources.get("allocated_gpus") or []

    gpu_models = "; ".join(
        sorted({str(gpu.get("name")) for gpu in gpu_inventory if gpu.get("name")})
    )
    gpu_visible_indices = "; ".join(
        str(gpu.get("index")) for gpu in gpu_inventory if gpu.get("index") is not None
    )
    gpu_uuids = "; ".join(
        str(gpu.get("uuid")) for gpu in gpu_inventory if gpu.get("uuid")
    )
    gpu_total_vram_mib = max(
        (float(gpu.get("memory_total_mib", 0.0) or 0.0) for gpu in gpu_inventory),
        default=None,
    )
    gpu_count = len(gpu_inventory)
    reconstruction_wall = timing.get("reconstruction_wall_seconds")
    profiling_wall = timing.get("profiling_wall_seconds")
    work_wall = reconstruction_wall if reconstruction_wall is not None else profiling_wall
    gpu_hours = (
        float(work_wall) / 3600.0 * gpu_count
        if isinstance(work_wall, (int, float)) and gpu_count
        else 0.0
    )
    profiling_input = metadata.get("input") or {}
    canonical_crops = metadata.get("canonical_crops") or {}

    return {
        "run_id": metadata.get("run_id"),
        "stage": metadata.get("stage", "reconstruction"),
        "corpus": metadata.get("corpus"),
        "backend": metadata.get("backend"),
        "adapter": metadata.get("adapter"),
        "status": metadata.get("status"),
        "tabulus_commit": git.get("commit"),
        "git_dirty": git.get("tracked_worktree_dirty"),
        "conda_env": metadata.get("environment", {}).get("conda_default_env"),
        "slurm_job_id": slurm.get("slurm_job_id"),
        "partition": slurm.get("slurm_job_partition"),
        "node": slurm.get("slurmd_nodename") or metadata.get("host", {}).get("hostname"),
        "slurm_gpu_ids": slurm.get("slurm_job_gpus"),
        "cuda_visible_devices": slurm.get("cuda_visible_devices"),
        "gpu_visible_indices": gpu_visible_indices,
        "gpu_uuids": gpu_uuids,
        "gpu_models": gpu_models,
        "gpu_count": gpu_count,
        "gpu_total_vram_mib": gpu_total_vram_mib,
        "peak_gpu_memory_mib": resources.get("gpu", {}).get("peak_gpu_memory_mib"),
        "max_host_rss_mib": resources.get("max_host_rss_mib"),
        "start_time_utc": timing.get("start_time_utc"),
        "end_time_utc": timing.get("end_time_utc"),
        "profiling_wall_seconds": profiling_wall,
        "reconstruction_wall_seconds": reconstruction_wall,
        "pdfs_requested": profiling_input.get("pdfs_requested"),
        "pages_total_known": profiling_input.get("pages_total_known"),
        "pages_complete": profiling_input.get("pages_complete"),
        "canonical_crop_roots": canonical_crops.get("paper_crop_roots"),
        "tables_detected": canonical_crops.get("tables_found"),
        "canonical_crops_saved": canonical_crops.get("crops_saved"),
        "batch_elapsed_seconds": reconstruction.get("batch_elapsed_seconds"),
        "median_seconds_per_table": runtime_stats.get("median"),
        "iqr_seconds_per_table": runtime_stats.get("iqr"),
        "p95_seconds_per_table": runtime_stats.get("p95"),
        "tables_per_minute": reconstruction.get("tables_per_minute"),
        "gpu_hours": gpu_hours,
        "tables_requested": reconstruction.get("tables_requested"),
        "tables_ok": reconstruction.get("tables_ok"),
        "tables_empty": reconstruction.get("tables_empty"),
        "tables_error": reconstruction.get("tables_error"),
        "prediction_csvs": reconstruction.get("prediction_csvs"),
        "structured_reconstruction_yield": reconstruction.get("structured_reconstruction_yield"),
        "parsed_zero": parsing.get("zero"),
        "parsed_one": parsing.get("one"),
        "parsed_multiple": parsing.get("multiple"),
        "failure_generation_ceiling": failures.get("generation_ceiling", 0),
        "failure_empty_no_structured_output": failures.get("empty_no_structured_output", 0),
        "failure_no_parseable_table": failures.get("no_parseable_table", 0),
        "failure_multiple_table_ambiguity": failures.get("multiple_table_ambiguity", 0),
        "failure_runtime_error": failures.get("runtime_error", 0),
        "successful_reference_like": downstream.get("successful_reference_like"),
        "successful_non_reference_like": downstream.get("successful_non_reference_like"),
        "inherited_continuation_decisions_all": downstream.get("inherited_continuation_decisions_all"),
        "classification_unavailable": downstream.get("unavailable_for_independent_classification"),
        "median_rows": (complexity.get("rows") or {}).get("median"),
        "median_columns": (complexity.get("columns") or {}).get("median"),
        "median_cells": (complexity.get("cells") or {}).get("median"),
        "adapter_version": provenance.get("adapter_version"),
        "model_version": provenance.get("model_version"),
        "run_dir": metadata.get("paths", {}).get("run_dir"),
    }
