from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from evaluation.jvsta.common import (
    GpuSampler,
    git_metadata,
    host_identity,
    load_json_object,
    query_gpu_inventory,
    select_allocated_gpus,
    slurm_metadata,
    utc_now_iso,
    write_json,
)
from evaluation.jvsta.run_experiment import _run_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one publication-grade Tabulus PDF-profiling experiment while "
            "recording execution, hardware, timing, resource, PDF, and canonical "
            "table-crop metadata."
        )
    )
    parser.add_argument("--corpus", required=True, help="Short corpus label, e.g. ald or asd.")
    parser.add_argument(
        "--pdf-folder",
        required=True,
        type=Path,
        help="Folder containing the corpus PDFs directly (non-recursive).",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("pipeline", "hybrid-engine"),
        help="MinerU profiling backend.",
    )
    parser.add_argument("--method", choices=("auto", "txt", "ocr"), default="auto")
    parser.add_argument("--effort", choices=("medium", "high"), default="high")
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="Experiment output root, preferably outside the Git repository.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=1.0,
        help="Seconds between nvidia-smi samples for GPU-backed profiling.",
    )
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_id(corpus: str, backend: str) -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    job_id = os.environ.get("SLURM_JOB_ID")
    suffix = f"slurm-{job_id}" if job_id else f"pid-{os.getpid()}"
    return f"{timestamp}__{corpus}__mineru-{backend}__{suffix}"


def _pdf_page_count(pdf_path: Path) -> int | None:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        return None
    completed = subprocess.run(
        [pdfinfo, str(pdf_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def summarize_pdf_inputs(pdf_folder: Path) -> dict[str, Any]:
    pdfs = sorted(path for path in Path(pdf_folder).iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    page_counts = {path.name: _pdf_page_count(path) for path in pdfs}
    known_pages = [value for value in page_counts.values() if isinstance(value, int)]
    return {
        "pdfs_requested": len(pdfs),
        "pdf_names": [path.name for path in pdfs],
        "page_counts": page_counts,
        "pages_known_for_pdfs": len(known_pages),
        "pages_total_known": sum(known_pages),
        "pages_complete": len(known_pages) == len(pdfs),
    }


def summarize_canonical_crops(crops_root: Path) -> dict[str, Any]:
    index_paths = sorted(Path(crops_root).glob("*/tables_index.json"))
    papers: list[dict[str, Any]] = []
    tables_found = 0
    crops_saved = 0
    for path in index_paths:
        payload = load_json_object(path)
        tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []
        found = int(payload.get("tables_found", len(tables)) or 0)
        saved = int(payload.get("crops_saved", len(tables)) or 0)
        tables_found += found
        crops_saved += saved
        papers.append(
            {
                "paper": path.parent.name,
                "tables_index": str(path),
                "tables_found": found,
                "crops_saved": saved,
                "refs_start_page": payload.get("refs_start_page"),
            }
        )
    return {
        "paper_crop_roots": len(index_paths),
        "tables_found": tables_found,
        "crops_saved": crops_saved,
        "papers": papers,
    }


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
    if not args.pdf_folder.is_dir():
        raise SystemExit(f"PDF folder does not exist: {args.pdf_folder}")

    input_summary = summarize_pdf_inputs(args.pdf_folder)
    if input_summary["pdfs_requested"] == 0:
        raise SystemExit(f"No PDF files found directly inside: {args.pdf_folder}")

    run_id = args.run_id or _run_id(args.corpus, args.backend)
    run_dir = args.run_root.resolve() / "profiling" / args.corpus / args.backend / run_id
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    profiler_root = run_dir / "mineru-output"
    crops_root = run_dir / "table-crops"
    metadata_path = run_dir / "run_metadata.json"
    stdout_path = run_dir / "profiling.stdout.log"
    stderr_path = run_dir / "profiling.stderr.log"
    resource_path = run_dir / "profiling.time.txt"
    gpu_samples = run_dir / "gpu_samples.csv"

    slurm = slurm_metadata()
    inventory = query_gpu_inventory()
    gpu_backed = args.backend == "hybrid-engine"
    export_canonical_crops = args.backend == "hybrid-engine"
    allocated_gpus = (
        select_allocated_gpus(
            inventory,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
        )
        if gpu_backed
        else []
    )

    start_time_utc = utc_now_iso()
    total_start = time.perf_counter()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "stage": "profiling",
        "run_id": run_id,
        "corpus": args.corpus,
        "profiler": "mineru",
        "backend": args.backend,
        "method": args.method,
        "effort": args.effort,
        "canonical_crop_export": export_canonical_crops,
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
            "gpu_inventory": inventory if gpu_backed else [],
        },
        "input": input_summary,
        "timing": {"start_time_utc": start_time_utc},
        "paths": {
            "run_dir": str(run_dir),
            "pdf_folder": str(args.pdf_folder.resolve()),
            "profiler_root": str(profiler_root),
            "crops_root": str(crops_root),
            "run_metadata": str(metadata_path),
            "gpu_samples": str(gpu_samples),
        },
    }
    write_json(metadata_path, metadata)

    sampler = GpuSampler(gpu_samples, interval_seconds=args.gpu_sample_interval)
    if gpu_backed:
        sampler.start()

    command = [
        sys.executable,
        "-m",
        "tabulus.cli",
        "profile",
        "--folder",
        str(args.pdf_folder.resolve()),
        "--backend",
        args.backend,
        "--method",
        args.method,
        "--effort",
        args.effort,
        "--out",
        str(profiler_root),
    ]
    if export_canonical_crops:
        command.extend(["--table-crops-out", str(crops_root)])
    else:
        command.append("--no-export-table-crops")

    try:
        result = _run_command(
            command,
            cwd=repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            resource_path=resource_path,
        )
        metadata["commands"] = {"profiling": result}
        command_ok = result["return_code"] == 0
        if export_canonical_crops:
            metadata["canonical_crops"] = {
                "export_enabled": True,
                **summarize_canonical_crops(crops_root),
            }
            expected_papers = int(input_summary["pdfs_requested"])
            crop_roots = int(metadata["canonical_crops"]["paper_crop_roots"])
            metadata["status"] = "success" if command_ok and crop_roots == expected_papers else "failed"
            if command_ok and crop_roots != expected_papers:
                metadata["harness_error"] = (
                    f"Profiling returned success but produced {crop_roots} canonical crop roots "
                    f"for {expected_papers} PDFs."
                )
        else:
            metadata["canonical_crops"] = {
                "export_enabled": False,
                "paper_crop_roots": 0,
                "tables_found": 0,
                "crops_saved": 0,
                "papers": [],
            }
            metadata["status"] = "success" if command_ok else "failed"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["harness_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        gpu_resource = sampler.stop() if gpu_backed else {
            "peak_gpu_memory_mib": None,
            "peak_gpu_memory_by_uuid_mib": {},
            "measurement": "not sampled for CPU pipeline backend",
            "sample_interval_seconds": None,
        }
        metadata["resources"]["gpu"] = gpu_resource
        command_metadata = (metadata.get("commands") or {}).get("profiling") or {}
        resource_usage = command_metadata.get("resource_usage") or {}
        metadata["resources"]["max_host_rss_mib"] = resource_usage.get("max_rss_mib")
        metadata["timing"]["end_time_utc"] = utc_now_iso()
        metadata["timing"]["total_wall_seconds"] = time.perf_counter() - total_start
        metadata["timing"]["profiling_wall_seconds"] = command_metadata.get("wall_seconds")
        metadata["git_end"] = git_metadata(repo_root)
        write_json(metadata_path, metadata)

    print(f"Run metadata: {metadata_path}")
    print(f"Run status: {metadata['status']}")
    print(f"PDFs requested: {input_summary['pdfs_requested']}")
    print(f"Canonical crop export: {'enabled' if export_canonical_crops else 'disabled'}")
    crops = metadata.get("canonical_crops") or {}
    print(f"Canonical crop roots: {crops.get('paper_crop_roots')}")
    print(f"Tables found: {crops.get('tables_found')}")
    print(f"Crops saved: {crops.get('crops_saved')}")
    return 0 if metadata["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
