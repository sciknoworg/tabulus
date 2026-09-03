from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

SSH_HOST = "gwdg"

REMOTE_BASE = "/user/v.rumleanschi/u25807/NuExtract3"

REMOTE_INPUT = f"{REMOTE_BASE}/work/input"
REMOTE_OUTPUT = f"{REMOTE_BASE}/work/output"

REMOTE_SUBMIT = f"{REMOTE_BASE}/png_to_csv_nuextract3_sbatch.sh"


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=True,
    )


def ssh(cmd: str) -> subprocess.CompletedProcess[str]:
    return run_cmd(["ssh", SSH_HOST, cmd])


def scp_to(local: Path, remote: str) -> None:
    run_cmd(["scp", str(local), f"{SSH_HOST}:{remote}"])


def submit(remote_in_png: str, remote_out_csv: str) -> str:
    completed = ssh(
        f"bash {REMOTE_SUBMIT} {remote_in_png} {remote_out_csv}"
    )

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")

    match = re.search(
        r"Submitted batch job\s+(\d+)",
        output,
    )

    if not match:
        raise RuntimeError(
            f"Could not parse Slurm JobID.\n{output}"
        )

    return match.group(1)


def detect_paper_id(local_png: Path) -> str:
    for part in local_png.parts:
        if re.fullmatch(r"P\d+", part):
            return part

    raise RuntimeError(
        f"Could not detect paper ID like P51 from path: {local_png}"
    )


def run_nuextract3(local_png: Path) -> None:
    local_png = Path(local_png).resolve()

    if not local_png.exists():
        raise FileNotFoundError(f"PNG not found: {local_png}")

    if local_png.suffix.lower() != ".png":
        raise ValueError(f"Expected PNG file, got: {local_png.suffix}")

    stem = local_png.stem
    paper_id = detect_paper_id(local_png)

    remote_in_png = f"{REMOTE_INPUT}/{paper_id}_{stem}.png"

    remote_paper_dir = f"{REMOTE_OUTPUT}/{paper_id}"
    remote_out_csv = f"{remote_paper_dir}/{stem}.csv"

    ssh(f"mkdir -p {REMOTE_INPUT} {remote_paper_dir}")

    print("[NuExtract3] Upload PNG")
    print(f"[NuExtract3] Paper ID: {paper_id}")
    print(f"[NuExtract3] Remote input: {remote_in_png}")
    print(f"[NuExtract3] Remote output: {remote_out_csv}")

    scp_to(local_png, remote_in_png)

    print("[NuExtract3] Submit Slurm job")

    job_id = submit(
        remote_in_png,
        remote_out_csv,
    )

    print(f"[NuExtract3] Submitted JobID: {job_id}")
    print("[NuExtract3] Remote CSV:")
    print(remote_out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a PNG to KISSKI and submit NuExtract3 CSV extraction job."
    )

    parser.add_argument(
        "--png",
        required=True,
        help="Local path to input table PNG.",
    )

    args = parser.parse_args()

    run_nuextract3(Path(args.png))