from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.jvsta.common import flatten_run_metadata, load_json_object


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate JVSTA run_metadata.json files into one analysis-ready CSV."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata_paths = sorted(args.run_root.rglob("run_metadata.json"))
    if not metadata_paths:
        raise SystemExit(f"No run_metadata.json files found below {args.run_root}")

    rows = [flatten_run_metadata(load_json_object(path)) for path in metadata_paths]
    fieldnames = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Runs summarized: {len(rows)}")
    print(f"CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
