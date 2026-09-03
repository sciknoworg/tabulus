from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from evaluation.jvsta.common import git_metadata


FALSE_VALUES = {"", "0", "false", "no", "off"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render or submit JVSTA Slurm jobs from a CSV manifest. Dry-run is "
            "the default; pass --submit to call sbatch."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--only-stage", choices=("profiling", "reconstruction"), default=None)
    parser.add_argument("--only-corpus", default=None)
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _expand(value: str | None) -> str:
    if value is None:
        return ""
    return os.path.expanduser(os.path.expandvars(value.strip()))


def _enabled(value: str | None) -> bool:
    return str(value or "1").strip().lower() not in FALSE_VALUES


def _required(row: dict[str, str], name: str) -> str:
    value = _expand(row.get(name))
    if not value:
        raise ValueError(f"Manifest row is missing required field {name!r}: {row}")
    return value


def _safe_job_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)[:40]


def build_sbatch_command(
    row: dict[str, str],
    *,
    repo_root: Path,
    expected_commit: str,
) -> list[str]:
    stage = _required(row, "stage").lower()
    if stage not in {"profiling", "reconstruction"}:
        raise ValueError(f"Unsupported stage {stage!r}")

    corpus = _required(row, "corpus")
    conda_env = _required(row, "conda_env")
    partition = _required(row, "partition")
    cpus = _required(row, "cpus")
    memory = _required(row, "memory")
    time_limit = _required(row, "time_limit")
    input_path = _required(row, "input_path")
    run_root = _required(row, "run_root")
    gres = _expand(row.get("gres"))
    nodelist = _expand(row.get("nodelist"))

    logs_dir = Path(run_root) / "slurm-logs"
    if stage == "profiling":
        backend = _required(row, "backend")
        label = f"profile-{corpus}-{backend}"
        payload = [
            "python",
            "-m",
            "evaluation.jvsta.run_profiling_experiment",
            "--corpus",
            corpus,
            "--pdf-folder",
            input_path,
            "--backend",
            backend,
            "--method",
            _expand(row.get("method")) or "auto",
            "--effort",
            _expand(row.get("effort")) or "high",
            "--run-root",
            run_root,
            "--expected-commit",
            expected_commit,
        ]
    else:
        adapter = _required(row, "adapter")
        label = f"recon-{corpus}-{adapter}"
        payload = [
            "python",
            "-m",
            "evaluation.jvsta.run_experiment",
            "--corpus",
            corpus,
            "--crops-folder",
            input_path,
            "--adapter",
            adapter,
            "--device",
            _expand(row.get("device")) or "gpu:0",
            "--run-root",
            run_root,
            "--expected-commit",
            expected_commit,
        ]

    job_name = _safe_job_token(label)
    command = [
        "sbatch",
        "-p",
        partition,
        "--cpus-per-task",
        cpus,
        "--mem",
        memory,
        "--time",
        time_limit,
        "-J",
        job_name,
        "-o",
        str(logs_dir / f"{job_name}-%j.out"),
    ]
    if gres:
        command.extend(["--gres", gres])
    if nodelist:
        command.extend(["--nodelist", nodelist])

    command.extend(
        [
            str(repo_root / "evaluation" / "jvsta" / "slurm" / "run_job.sh"),
            str(repo_root),
            conda_env,
            *payload,
        ]
    )
    return command


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return [dict(row) for row in reader]


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    git = git_metadata(repo_root)
    if git["tracked_worktree_dirty"]:
        raise SystemExit("Refusing to submit publication jobs from a dirty tracked worktree.")

    expected_commit = args.expected_commit or str(git["commit"])
    if not str(git["commit"]).startswith(expected_commit):
        raise SystemExit(
            f"Current commit {git['commit']} does not match --expected-commit {expected_commit}."
        )

    rows = _read_manifest(args.manifest)
    selected: list[dict[str, str]] = []
    for row in rows:
        if not _enabled(row.get("enabled")):
            continue
        if args.only_stage and _expand(row.get("stage")).lower() != args.only_stage:
            continue
        if args.only_corpus and _expand(row.get("corpus")) != args.only_corpus:
            continue
        selected.append(row)

    if not selected:
        raise SystemExit("No enabled manifest rows matched the requested filters.")

    submitted = 0
    for row in selected:
        command = build_sbatch_command(row, repo_root=repo_root, expected_commit=expected_commit)
        print(shlex.join(command))
        if not args.submit:
            continue
        log_path = Path(command[command.index("-o") + 1])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        print(completed.stdout.strip())
        submitted += 1

    if args.submit:
        print(f"Jobs submitted: {submitted}")
    else:
        print(f"Dry-run rows rendered: {len(selected)}")
        print("No jobs were submitted. Re-run with --submit after reviewing the commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
