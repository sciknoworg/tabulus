from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

TARGET_FOLDER_NAME = "minerU_extracted_table.png"


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def find_pngs(root_dir: Path) -> list[Path]:
    pngs: list[Path] = []

    for folder in root_dir.rglob(TARGET_FOLDER_NAME):
        if folder.is_dir():
            pngs.extend(sorted(folder.glob("*.png")))

    return pngs


def run_runner(runner_path: Path, png_path: Path) -> None:
    print("=" * 80)
    print(f"[NuExtract3 Batch] PNG: {png_path}")

    cmd = [
        "python",
        str(runner_path),
        "--png",
        str(png_path),
    ]

    run_cmd(cmd)

    print(f"[NuExtract3 Batch] Submitted: {png_path.name}")


def main(root_dir: Path, runner_path: Path) -> None:
    root_dir = root_dir.resolve()
    runner_path = runner_path.resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")

    if not runner_path.exists():
        raise FileNotFoundError(f"Runner not found: {runner_path}")

    pngs = find_pngs(root_dir)

    print(f"[NuExtract3 Batch] PNG files found: {len(pngs)}")

    if not pngs:
        return

    for png_path in pngs:
        try:
            run_runner(runner_path, png_path)
        except Exception as error:
            print(f"[NuExtract3 Batch] FAILED: {png_path}")
            print(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NuExtract3 remote runner for all MinerU PNG tables."
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Root dataset directory.",
    )

    parser.add_argument(
        "--runner",
        required=True,
        help="Path to nuextract3_remote_runner.py",
    )

    args = parser.parse_args()

    main(
        root_dir=Path(args.root),
        runner_path=Path(args.runner),
    )