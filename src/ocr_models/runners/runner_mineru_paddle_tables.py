from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from runners.services.mineru_tables_png_runner import run as mineru_tables_png_run

PADDLE_BASE_URL = "http://127.0.0.1:8002"
PADDLE_IMAGES_API = f"{PADDLE_BASE_URL}/ocr/images"
PADDLE_HEALTH_URL = f"{PADDLE_BASE_URL}/health"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _paddle_is_up() -> bool:
    try:
        response = requests.get(PADDLE_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def run(pdf_path: Path, out_dir: Path) -> None:
    start_time = time.perf_counter()

    pdf_path = Path(pdf_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[MinerU + PaddleOCR-VL] Step 1: MinerU table crops", flush=True)
    mineru_tables_png_run(pdf_path, out_dir)

    crops_dir = out_dir / "images" / "tables"
    index_path = crops_dir / "tables_index.json"

    if not index_path.exists():
        raise RuntimeError("MinerU did not create images/tables/tables_index.json")

    index_data = json.loads(index_path.read_text(encoding="utf-8"))

    tables = index_data.get("tables") if isinstance(index_data, dict) else []
    if not isinstance(tables, list):
        tables = []

    refs_start_page = index_data.get("refs_start_page") if isinstance(index_data, dict) else None
    if not isinstance(refs_start_page, int):
        refs_start_page = None

    png_paths: List[Path] = []

    for table in tables:
        png = table.get("png")
        if isinstance(png, str):
            png_path = Path(png)
            if png_path.exists():
                png_paths.append(png_path)

    if not png_paths:
        _write_json(
            out_dir / "tables.json",
            {
                "pdf": pdf_path.name,
                "tables_found": 0,
                "tables": [],
                "refs_start_page": refs_start_page,
            },
        )
        print("[MinerU + PaddleOCR-VL] No tables found", flush=True)
        return

    if not _paddle_is_up():
        elapsed = time.perf_counter() - start_time

        _write_json(
            out_dir / "tables.json",
            {
                "pdf": pdf_path.name,
                "mineru_crops": len(png_paths),
                "tables_found": 0,
                "tables": [],
                "note": "PaddleOCR-VL not reachable. Only MinerU PNG crops were produced.",
                "refs_start_page": refs_start_page,
            },
        )

        (out_dir / "notes.md").write_text(
            "\n".join(
                [
                    "- Library: MinerU + PaddleOCR-VL",
                    f"- PDF: {pdf_path.name}",
                    f"- mineru_crops: {len(png_paths)}",
                    "- PaddleOCR-VL: skipped, health check failed",
                    f"- refs_start_page: {refs_start_page}",
                    f"- Duration: {elapsed:.2f} s",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        print("[MinerU + PaddleOCR-VL] PaddleOCR-VL not reachable. Skipped.", flush=True)
        return

    print("[MinerU + PaddleOCR-VL] Step 2: PaddleOCR-VL on table crops", flush=True)

    files = [
        (
            "files",
            (
                png_path.name,
                png_path.read_bytes(),
                "image/png",
            ),
        )
        for png_path in png_paths
    ]

    response = requests.post(
        PADDLE_IMAGES_API,
        files=files,
        headers={"Expect": ""},
        timeout=(10, 60 * 60),
        proxies={"http": None, "https": None},
    )

    print("[MinerU + PaddleOCR-VL] Paddle status:", response.status_code, flush=True)
    response.raise_for_status()

    payload: Dict[str, Any] = response.json()
    _write_json(out_dir / "raw_paddle_vl_images_response.json", payload)

    if not payload.get("success"):
        raise RuntimeError("PaddleOCR-VL /ocr/images returned success=false")

    paddle_tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []

    mineru_by_name = {
        table.get("png_name"): table
        for table in tables
        if isinstance(table, dict) and table.get("png_name")
    }

    merged_tables: List[Dict[str, Any]] = []

    for table in paddle_tables:
        if not isinstance(table, dict):
            continue

        source_file = table.get("source_file")
        mineru_meta = mineru_by_name.get(source_file) if isinstance(source_file, str) else None

        if mineru_meta:
            merged = dict(table)
            merged["mineru_page_nr"] = mineru_meta.get("page_nr")
            merged["mineru_bbox"] = mineru_meta.get("bbox")
            merged["mineru_png_path"] = mineru_meta.get("png")
            merged["in_references"] = mineru_meta.get("in_references")
            merged_tables.append(merged)
        else:
            merged_tables.append(table)

    output = {
        "pdf": pdf_path.name,
        "mineru_crops": len(png_paths),
        "tables_found": len(
            [table for table in merged_tables if (table.get("n_rows") or 0) > 0]
        ),
        "tables": merged_tables,
        "refs_start_page": refs_start_page,
    }

    _write_json(out_dir / "tables.json", output)

    elapsed = time.perf_counter() - start_time

    (out_dir / "notes.md").write_text(
        "\n".join(
            [
                "- Library: MinerU + PaddleOCR-VL",
                f"- PDF: {pdf_path.name}",
                f"- mineru_crops: {len(png_paths)}",
                f"- tables_found: {output['tables_found']}",
                f"- refs_start_page: {refs_start_page}",
                f"- Duration: {elapsed:.2f} s",
                "- Files: images/tables/*.png, images/tables/tables_index.json, tables.json, raw_paddle_vl_images_response.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("[MinerU + PaddleOCR-VL] Done.", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MinerU crops and PaddleOCR-VL table OCR.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF.")
    parser.add_argument("--out", required=True, help="Output directory.")

    args = parser.parse_args()

    run(
        pdf_path=Path(args.pdf),
        out_dir=Path(args.out),
    )