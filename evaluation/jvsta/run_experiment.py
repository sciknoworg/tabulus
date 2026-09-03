from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from evaluation.jvsta.common import (
    GpuSampler,
    collect_model_provenance,
    git_metadata,
    host_identity,
    load_json_object,
    parse_gnu_time_verbose,
    query_gpu_inventory,
    select_allocated_gpus,
    slurm_metadata,
    summarize_batch_summaries,
    summarize_classification_manifests,
    summarize_output_complexity,
    utc_now_iso,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one publication-grade Tabulus table-reconstruction experiment "
            "while recording execution, hardware, timing, resource, coverage, "
            "parsing, robustness, and downstream-classification metadata."
        )
    )
    parser.add_argument("--corpus", required=True, help="Short corpus label, e.g. ald or asd.")
    parser.add_argument(
        "--crops-folder",
        required=True,
        type=Path,
        help="Folder whose immediate child directories are canonical crop roots.",
    )
    parser.add_argument("--adapter", required=True, help="Registered Tabulus table OCR adapter name.")
    parser.add_argument("--device", default="gpu:0", help="Device passed to the Tabulus adapter.")
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="Experiment output root. Prefer a location outside the Git repository.",
    )
    parser.add_argument("--run-id", default=None, help="Optional explicit unique run identifier.")
    parser.add_argument(
        "--expected-commit",
        default=None,
        help="Optional Git SHA/prefix that the current Tabulus checkout must match.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow tracked working-tree modifications. Publication runs should normally omit this.",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip downstream reference-table classification.",
    )
    parser.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=1.0,
        help="Seconds between nvidia-smi memory samples (default: 1.0).",
    )
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_id(corpus: str, adapter: str) -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    job_id = os.environ.get("SLURM_JOB_ID")
    suffix = f"slurm-{job_id}" if job_id else f"pid-{os.getpid()}"
    safe_adapter = adapter.replace("/", "-")
    return f"{timestamp}__{corpus}__{safe_adapter}__{suffix}"


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    resource_path: Path,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    gnu_time = Path("/usr/bin/time")
    wrapped = command
    if gnu_time.is_file():
        wrapped = [str(gnu_time), "-v", "-o", str(resource_path), *command]

    start_iso = utc_now_iso()
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            wrapped,
            cwd=cwd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - start
    end_iso = utc_now_iso()
    resource = parse_gnu_time_verbose(resource_path) if gnu_time.is_file() else {}
    return {
        "command": command,
        "command_shell": shlex.join(command),
        "return_code": completed.returncode,
        "start_time_utc": start_iso,
        "end_time_utc": end_iso,
        "wall_seconds": elapsed,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "resource_usage": resource,
    }


def _discover_batch_summaries(reconstruction_root: Path) -> list[Path]:
    return sorted(reconstruction_root.rglob("batch_summary.json"))


def _classification_paths(summary_paths: list[Path]) -> list[Path]:
    return [path.parent / "reference_table_classification.json" for path in summary_paths]


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    git = git_metadata(repo_root)

    if git["tracked_worktree_dirty"] and not args.allow_dirty:
        raise SystemExit(
            "Refusing publication run because tracked Git files are modified. "
            "Commit/stash changes or pass --allow-dirty explicitly."
        )
    if args.expected_commit and not str(git["commit"]).startswith(args.expected_commit):
        raise SystemExit(
            f"Tabulus commit mismatch: expected {args.expected_commit!r}, found {git['commit']}."
        )
    if not args.crops_folder.is_dir():
        raise SystemExit(f"Canonical crops folder does not exist: {args.crops_folder}")

    run_id = args.run_id or _run_id(args.corpus, args.adapter)
    run_dir = args.run_root.resolve() / args.corpus / args.adapter / run_id
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    reconstruction_root = run_dir / "reconstruction"
    metadata_path = run_dir / "run_metadata.json"
    reconstruction_stdout = run_dir / "reconstruction.stdout.log"
    reconstruction_stderr = run_dir / "reconstruction.stderr.log"
    reconstruction_resource = run_dir / "reconstruction.time.txt"
    classification_stdout = run_dir / "classification.stdout.log"
    classification_stderr = run_dir / "classification.stderr.log"
    classification_resource = run_dir / "classification.time.txt"
    reconstruction_list = run_dir / "reconstruction_dirs.txt"
    gpu_samples = run_dir / "gpu_samples.csv"

    slurm = slurm_metadata()
    inventory = query_gpu_inventory()
    allocated_gpus = select_allocated_gpus(
        inventory,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
    )

    start_time_utc = utc_now_iso()
    total_start = time.perf_counter()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "stage": "reconstruction",
        "run_id": run_id,
        "corpus": args.corpus,
        "adapter": args.adapter,
        "device": args.device,
        "status": "running",
        "git": git,
        "host": host_identity(),
        "slurm": slurm,
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "resources": {
            "allocated_gpus": allocated_gpus,
            "gpu_inventory": inventory,
        },
        "timing": {"start_time_utc": start_time_utc},
        "paths": {
            "run_dir": str(run_dir),
            "crops_folder": str(args.crops_folder.resolve()),
            "reconstruction_root": str(reconstruction_root),
            "run_metadata": str(metadata_path),
            "gpu_samples": str(gpu_samples),
        },
    }
    write_json(metadata_path, metadata)

    sampler = GpuSampler(gpu_samples, interval_seconds=args.gpu_sample_interval)
    sampler.start()

    reconstruction_command = [
        sys.executable,
        "-m",
        "tabulus.cli",
        "reconstruct-tables",
        "--crops-folder",
        str(args.crops_folder.resolve()),
        "--adapter",
        args.adapter,
        "--device",
        args.device,
        "--out",
        str(reconstruction_root),
    ]

    classification_result: dict[str, Any] | None = None
    try:
        reconstruction_result = _run_command(
            reconstruction_command,
            cwd=repo_root,
            stdout_path=reconstruction_stdout,
            stderr_path=reconstruction_stderr,
            resource_path=reconstruction_resource,
        )
        metadata["commands"] = {"reconstruction": reconstruction_result}
        write_json(metadata_path, metadata)

        summary_paths = _discover_batch_summaries(reconstruction_root)
        if reconstruction_result["return_code"] == 0 and not summary_paths:
            raise RuntimeError("Reconstruction command succeeded but produced no batch_summary.json files.")

        if reconstruction_result["return_code"] == 0 and not args.skip_classification:
            reconstruction_dirs = [path.parent.resolve() for path in summary_paths]
            reconstruction_list.write_text(
                "\n".join(str(path) for path in reconstruction_dirs) + "\n",
                encoding="utf-8",
            )
            classification_command = [
                sys.executable,
                "-m",
                "tabulus.cli",
                "classify-reference-tables",
                "--reconstruction-list",
                str(reconstruction_list),
                "--adapter",
                args.adapter,
            ]
            classification_result = _run_command(
                classification_command,
                cwd=repo_root,
                stdout_path=classification_stdout,
                stderr_path=classification_stderr,
                resource_path=classification_resource,
            )
            metadata["commands"]["classification"] = classification_result

        metadata["reconstruction"] = summarize_batch_summaries(summary_paths)
        metadata["output_complexity"] = summarize_output_complexity(summary_paths)
        metadata["model_provenance"] = collect_model_provenance(summary_paths)

        classification_paths = [
            path for path in _classification_paths(summary_paths) if path.is_file()
        ]
        metadata["downstream"] = (
            summarize_classification_manifests(classification_paths)
            if classification_paths
            else None
        )

        reconstruction_ok = reconstruction_result["return_code"] == 0
        classification_ok = (
            args.skip_classification
            or (
                classification_result is not None
                and classification_result["return_code"] == 0
            )
        )
        metadata["status"] = "success" if reconstruction_ok and classification_ok else "failed"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["harness_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        gpu_resource = sampler.stop()
        metadata.setdefault("resources", {})["gpu"] = gpu_resource
        host_rss_values = []
        for command_metadata in (metadata.get("commands") or {}).values():
            value = (command_metadata.get("resource_usage") or {}).get("max_rss_mib")
            if isinstance(value, (int, float)):
                host_rss_values.append(float(value))
        metadata["resources"]["max_host_rss_mib"] = max(host_rss_values, default=None)
        metadata["timing"]["end_time_utc"] = utc_now_iso()
        metadata["timing"]["total_wall_seconds"] = time.perf_counter() - total_start
        reconstruction_metadata = (metadata.get("commands") or {}).get("reconstruction") or {}
        metadata["timing"]["reconstruction_wall_seconds"] = reconstruction_metadata.get("wall_seconds")
        classification_metadata = (metadata.get("commands") or {}).get("classification") or {}
        metadata["timing"]["classification_wall_seconds"] = classification_metadata.get("wall_seconds")
        metadata["git_end"] = git_metadata(repo_root)
        write_json(metadata_path, metadata)

    print(f"Run metadata: {metadata_path}")
    print(f"Run status: {metadata['status']}")
    reconstruction = metadata.get("reconstruction") or {}
    if reconstruction:
        print(f"Tables requested: {reconstruction.get('tables_requested')}")
        print(f"Tables ok: {reconstruction.get('tables_ok')}")
        print(f"Tables empty: {reconstruction.get('tables_empty')}")
        print(f"Tables error: {reconstruction.get('tables_error')}")
        print(f"Prediction CSVs: {reconstruction.get('prediction_csvs')}")
    return 0 if metadata["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
