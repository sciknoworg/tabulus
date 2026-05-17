from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

SSH_HOST = "gwdg"

REMOTE_BASE = "/user/v.rumleanschi/u25807/runners"
REMOTE_INPUT = f"{REMOTE_BASE}/work/input"
REMOTE_OUTPUT = f"{REMOTE_BASE}/work/output"
REMOTE_LOGS = f"{REMOTE_BASE}/logs"
REMOTE_SUBMIT = f"{REMOTE_BASE}/bin/png_to_md_sbatch.sh"

POLL_SECONDS = 5
TIMEOUT_SECONDS = 60 * 60


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def ssh(cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_cmd(["ssh", SSH_HOST, cmd], check=check)


def scp_to(local: Path, remote: str) -> None:
    run_cmd(["scp", str(local), f"{SSH_HOST}:{remote}"], check=True)


def scp_from(remote: str, local: Path, check: bool = True) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["scp", f"{SSH_HOST}:{remote}", str(local)], check=check)


def submit(remote_in_png: str, remote_out_md: str) -> str:
    completed = ssh(f"bash {REMOTE_SUBMIT} {remote_in_png} {remote_out_md}", check=True)

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    match = re.search(r"Submitted batch job\s+(\d+)", output)

    if not match:
        raise RuntimeError(f"Could not parse Slurm JobID.\n{output}")

    return match.group(1)


def squeue_has(job_id: str) -> bool:
    completed = ssh(f"squeue -j {job_id} -h", check=False)
    return bool((completed.stdout or "").strip())


def wait_done(job_id: str) -> None:
    start = time.time()

    while True:
        if not squeue_has(job_id):
            return

        if time.time() - start > TIMEOUT_SECONDS:
            raise TimeoutError(f"Timed out waiting for job {job_id}")

        time.sleep(POLL_SECONDS)


def md_table_to_csv(md_path: Path, csv_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()

    table_lines: list[str] = []
    in_table = False

    for line in lines:
        if "|" in line:
            table_lines.append(line.strip())
            in_table = True
        elif in_table:
            break

    if len(table_lines) < 2:
        raise ValueError("No markdown table found in file.")

    header = table_lines[0]
    rows = table_lines[2:] if len(table_lines) > 2 else []

    parsed_rows = []
    parsed_rows.append([cell.strip() for cell in header.split("|") if cell.strip()])

    for row in rows:
        parsed_rows.append([cell.strip() for cell in row.split("|") if cell.strip()])

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(parsed_rows)


def run_chandra(local_png: Path, out_dir: Path | None = None) -> None:
    local_png = Path(local_png).resolve()

    if not local_png.exists():
        raise FileNotFoundError(f"PNG not found: {local_png}")

    if local_png.suffix.lower() != ".png":
        raise ValueError(f"Expected PNG file, got: {local_png.suffix}")

    stem = local_png.stem

    if out_dir is None:
        out_dir = local_png.parent
    else:
        out_dir = Path(out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    local_md = out_dir / f"{stem}.md"
    local_csv = out_dir / f"{stem}.csv"

    remote_in_png = f"{REMOTE_INPUT}/{stem}.png"
    remote_out_md = f"{REMOTE_OUTPUT}/{stem}.md"

    print("[Chandra] Step 1: upload PNG", flush=True)
    scp_to(local_png, remote_in_png)

    print("[Chandra] Step 2: submit Slurm job", flush=True)
    job_id = submit(remote_in_png, remote_out_md)
    print(f"[Chandra] JobID: {job_id}", flush=True)

    print("[Chandra] Step 3: wait for job", flush=True)
    wait_done(job_id)

    print("[Chandra] Step 4: download markdown result", flush=True)
    scp_from(remote_out_md, local_md, check=False)

    if local_md.exists() and local_md.stat().st_size > 0:
        print(f"[Chandra] MD downloaded: {local_md}", flush=True)

        try:
            md_table_to_csv(local_md, local_csv)
            print(f"[Chandra] CSV created: {local_csv}", flush=True)
        except Exception as error:
            print(f"[Chandra] Could not convert MD to CSV: {error}", flush=True)

        return

    remote_log = f"{REMOTE_LOGS}/chandra_md_{job_id}.log"
    local_log = out_dir / f"{stem}_job_{job_id}.log"

    scp_from(remote_log, local_log, check=False)
    print(f"[Chandra] MD missing. Log saved if available: {local_log}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chandra OCR remotely via KISSKI/Slurm.")
    parser.add_argument("--png", required=True, help="Path to input table PNG.")
    parser.add_argument("--out", required=False, help="Output directory. Default: PNG parent folder.")

    args = parser.parse_args()

    run_chandra(
        local_png=Path(args.png),
        out_dir=Path(args.out) if args.out else None,
    )