from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def _write_notes(out_dir: Path, notes: list[str]) -> None:
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def run(pdf_path: Path, out_dir: Path, pages: str | None = None):
    """
    Run surya_table and convert its results.json into per-table CSVs.

    pages: optional Surya CLI page selector, e.g. "1,2,3" or "5" (depends on surya_table support).
           If None, run all pages.
    """
    tables_dir = out_dir / "extracted_tables"
    raw_dir = out_dir / "raw"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    notes = [
        "- Library: surya-ocr (surya_table)",
        f"- PDF: {pdf_path.name}",
        "- Attempts:",
    ]

    exe = shutil.which("surya_table")
    if not exe:
        msg = "surya_table not found. Install with: python -m pip install surya-ocr"
        notes.append(f"  - FAILED: {msg}")
        _write_notes(out_dir, notes)
        raise RuntimeError(msg)

    cmd = [exe, str(pdf_path), "--output_dir", str(raw_dir)]
    if pages:
        cmd.extend(["--pages", pages])

    print(f"[surya] Running: {' '.join(cmd)}")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        notes.append("  - FAILED: surya_table returned non-zero exit code")
        if proc.stderr:
            notes.append("  - stderr (truncated):")
            notes.append(proc.stderr.strip()[:2000])
        _write_notes(out_dir, notes)
        raise RuntimeError(proc.stderr or "surya_table failed")

    results_path = raw_dir / "results.json"
    if not results_path.exists():
        notes.append("  - FAILED: results.json not found in output_dir")
        _write_notes(out_dir, notes)
        raise RuntimeError("Surya did not produce results.json")

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Surya usually keys by filename stem; sometimes it differs.
    key = pdf_path.stem
    if key not in data:
        if len(data) == 1:
            key = next(iter(data.keys()))
            notes.append(f"  - NOTE: results.json key mismatch; used '{key}' instead of '{pdf_path.stem}'")
        else:
            notes.append(f"  - FAILED: Could not find key '{pdf_path.stem}' in results.json")
            _write_notes(out_dir, notes)
            raise RuntimeError("Unexpected results.json format")

    pages_data = data.get(key) or []
    if not isinstance(pages_data, list):
        notes.append("  - FAILED: Unexpected pages format (expected list)")
        _write_notes(out_dir, notes)
        raise RuntimeError("Unexpected results.json format: pages is not a list")

    total_tables = 0
    saved_tables = 0

    for page_obj in pages_data:
        if not isinstance(page_obj, dict):
            continue

        page_num = _safe_int(page_obj.get("page", 0), default=0)
        table_idx = _safe_int(page_obj.get("table_idx", 0), default=0)

        rows_meta = page_obj.get("rows") or []
        cols_meta = page_obj.get("cols") or []
        cells = page_obj.get("cells") or []

        if not rows_meta or not cols_meta:
            continue

        # Collect row/col ids (stable ordering)
        row_ids = sorted({_safe_int(r.get("row_id"), -1) for r in rows_meta if "row_id" in r and r.get("row_id") is not None})
        col_ids = sorted({_safe_int(c.get("col_id"), -1) for c in cols_meta if "col_id" in c and c.get("col_id") is not None})
        row_ids = [r for r in row_ids if r >= 0]
        col_ids = [c for c in col_ids if c >= 0]
        if not row_ids or not col_ids:
            continue

        total_tables += 1

        # map (row_id, col_id) -> text (keep longest if duplicates)
        cell_text: dict[tuple[int, int], str] = {}
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            if "row_id" not in cell or "col_id" not in cell:
                continue

            r_id = _safe_int(cell.get("row_id"), -1)
            c_id = _safe_int(cell.get("col_id"), -1)
            if r_id < 0 or c_id < 0:
                continue

            txt = (cell.get("text") or "").strip()
            prev = cell_text.get((r_id, c_id))
            if prev is None or len(txt) > len(prev):
                cell_text[(r_id, c_id)] = txt

        # Build grid
        grid: list[list[str]] = [
            [cell_text.get((r_id, c_id), "") for c_id in col_ids]
            for r_id in row_ids
        ]

        out_file = tables_dir / f"page_{page_num:03d}_table_{table_idx:02d}.csv"
        _write_csv(out_file, grid)
        saved_tables += 1

    notes.append(f"  - extracted_table_objects={total_tables}")
    notes.append(f"  - saved_csv_tables={saved_tables}")
    notes.append(f"  - raw_results: {results_path}")
    if pages:
        notes.append(f"  - pages_arg: {pages}")

    _write_notes(out_dir, notes)
    print(f"[surya] extracted={total_tables}, saved={saved_tables}, out={tables_dir}")
