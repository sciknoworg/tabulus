import re
import subprocess
import sys
from pathlib import Path

SSH_HOST = "gwdg"

REMOTE_BASE = "/user/v.rumleanschi/u25807/runners"
REMOTE_INPUT = f"{REMOTE_BASE}/work/input"
REMOTE_OUTPUT = f"{REMOTE_BASE}/work/output"
REMOTE_SUBMIT = f"{REMOTE_BASE}/bin/pdf_to_txt_sbatch.sh"


def run(cmd):
    cp = subprocess.run(cmd, text=True, capture_output=True)
    if cp.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"Return code: {cp.returncode}\n"
            f"STDOUT:\n{cp.stdout}\n"
            f"STDERR:\n{cp.stderr}"
        )
    return cp


def ssh(cmd: str):
    return run(["ssh", SSH_HOST, cmd])


def scp_to(local: Path, remote: str):
    return run(["scp", str(local), f"{SSH_HOST}:{remote}"])


def ensure_remote_dir(remote_dir: str):
    ssh(f'mkdir -p "{remote_dir}"')


def detect_group(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"P\d+", part, re.IGNORECASE):
            return part
    raise ValueError("No group like P1/P2 found in path")


def submit(remote_in_pdf: str, remote_out_txt: str, start_page: int) -> str:
    cp = ssh(
        f'bash {REMOTE_SUBMIT} "{remote_in_pdf}" "{remote_out_txt}" "{start_page}"'
    )
    out = (cp.stdout or "") + "\n" + (cp.stderr or "")
    m = re.search(r"Submitted batch job\s+(\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse JobID:\n{out}")
    return m.group(1)


def main():
    if len(sys.argv) != 3:
        print("Usage: python submit_pdf_rawtext_grouped_jobs.py <pdf-path> <start-page>")
        sys.exit(1)

    local_pdf = Path(sys.argv[1]).resolve()
    if not local_pdf.exists():
        print(f"File not found: {local_pdf}")
        sys.exit(1)

    if local_pdf.suffix.lower() != ".pdf":
        print(f"Expected a PDF file, got: {local_pdf}")
        sys.exit(1)

    try:
        start_page = int(sys.argv[2])
        if start_page < 1:
            raise ValueError
    except ValueError:
        print("start-page must be an integer >= 1")
        sys.exit(1)

    group = detect_group(local_pdf)
    stem = local_pdf.stem

    remote_input_dir = f"{REMOTE_INPUT}/{group}"
    remote_output_dir = f"{REMOTE_OUTPUT}/{group}"

    remote_in_pdf = f"{remote_input_dir}/{local_pdf.name}"
    remote_out_txt = f"{remote_output_dir}/{stem}.txt"

    print(f"PDF: {local_pdf}")
    print(f"Group: {group}")
    print(f"Start page: {start_page}")

    ensure_remote_dir(remote_input_dir)
    ensure_remote_dir(remote_output_dir)

    print("[1] Uploading PDF...")
    scp_to(local_pdf, remote_in_pdf)

    print("[2] Submitting job...")
    jobid = submit(remote_in_pdf, remote_out_txt, start_page)

    print(f"Submitted JobID: {jobid}")
    print("You can fetch results later")
    print(f"Expected TXT: {remote_out_txt}")


if __name__ == "__main__":
    main()