from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from service.mineru_tables_png_runner import run as mineru_tables_png_run

DEEPSEEK_BASE_URL = "http://127.0.0.1:8000"

DEEPSEEK_IMAGES_API = f"{DEEPSEEK_BASE_URL}/ocr/images"
DEEPSEEK_REFERENCES_API = f"{DEEPSEEK_BASE_URL}/ocr/references"
DEEPSEEK_HEALTH_URL = f"{DEEPSEEK_BASE_URL}/health"

EXTRACT_REFERENCES = False
EXTRACT_TABLES_WITH_DEEPSEEK = False

try:
    import fitz
except Exception:
    fitz = None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _deepseek_is_up() -> bool:
    try:
        response = requests.get(DEEPSEEK_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def _render_pdf_pages_to_png(
    pdf_path: Path,
    start_page_1based: int,
    out_dir: Path,
    dpi: int = 180,
    max_pages: int = 8,
) -> List[Path]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required. Install it with: pip install pymupdf")

    out_dir.mkdir(parents=True, exist_ok=True)

    document = fitz.open(str(pdf_path))

    try:
        page_count = document.page_count
        start_idx = max(0, int(start_page_1based) - 1)

        if start_idx >= page_count:
            return []

        end_idx = min(page_count, start_idx + max_pages)

        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)

        png_paths: List[Path] = []

        for page_idx in range(start_idx, end_idx):
            page = document.load_page(page_idx)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            page_nr = page_idx + 1
            png_path = out_dir / f"ref_page_{page_nr:03d}.png"

            pixmap.save(str(png_path))
            png_paths.append(png_path)

        return png_paths

    finally:
        document.close()


def _looks_like_bib(text: str) -> bool:
    text = (text or "").strip()

    if not text:
        return False

    if text.upper() == "NONE":
        return False

    lower = text.lower()

    bad_hits = sum(
        marker in lower
        for marker in [
            "times new roman",
            "calibri",
            "spacing",
            "bullet points",
            "conclusion",
        ]
    )

    if bad_hits >= 2:
        return False

    hits = 0
    hits += 1 if re.search(r"\b(19|20)\d{2}\b", text) else 0
    hits += 1 if "doi" in lower or "doi.org" in lower else 0
    hits += 1 if "et al" in lower else 0
    hits += 1 if re.search(r"\bvol\.\b|\bno\.\b|\bpp\.\b", lower) else 0
    hits += 1 if re.search(r"https?://", lower) else 0

    return hits >= 2


def _post_single_ref_page(png_path: Path) -> Dict[str, Any]:
    files = [
        (
            "files",
            (
                png_path.name,
                png_path.read_bytes(),
                "image/png",
            ),
        )
    ]

    response = requests.post(
        DEEPSEEK_REFERENCES_API,
        files=files,
        headers={"Expect": ""},
        timeout=(10, 60 * 60),
        proxies={"http": None, "https": None},
    )

    response.raise_for_status()
    return response.json()


def _extract_references_page_by_page(
    pdf_path: Path,
    refs_start_page: int,
    out_dir: Path,
    dpi: int = 180,
    max_pages: int = 8,
    try_offsets: bool = True,
) -> Dict[str, Any]:
    pages_dir = out_dir / "images" / "references" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    candidates = [refs_start_page]

    if try_offsets:
        candidates = [
            refs_start_page,
            refs_start_page + 1,
            max(1, refs_start_page - 1),
        ]

    best_result: Optional[Dict[str, Any]] = None

    for start_page in candidates:
        rendered_pages = _render_pdf_pages_to_png(
            pdf_path=pdf_path,
            start_page_1based=start_page,
            out_dir=pages_dir,
            dpi=dpi,
            max_pages=max_pages,
        )

        if not rendered_pages:
            continue

        references: List[str] = []
        pages_sent = 0
        pages_debug: List[Dict[str, Any]] = []

        for png_path in rendered_pages:
            payload = _post_single_ref_page(png_path)

            refs = payload.get("references") or []

            pages_debug.append(
                {
                    "png": png_path.name,
                    "is_references": payload.get("is_references"),
                    "references_found": payload.get("references_found"),
                    "combined_text_snip": (payload.get("combined_text") or "")[:400],
                }
            )

            if isinstance(refs, list) and len(refs) == 1 and str(refs[0]).strip().upper() == "NONE":
                break

            joined_refs = "\n\n".join(str(ref) for ref in refs if str(ref).strip())

            if not _looks_like_bib(joined_refs):
                break

            references.extend(str(ref) for ref in refs if str(ref).strip())
            pages_sent += 1

        result = {
            "pdf": pdf_path.name,
            "refs_start_page_used": start_page,
            "refs_start_page_reported": refs_start_page,
            "pages_sent": pages_sent,
            "references_found": len(references),
            "references": references,
            "pages_debug": pages_debug,
        }

        if best_result is None or result["references_found"] > best_result["references_found"]:
            best_result = result

        if result["references_found"] >= 5:
            break

    if best_result is None:
        return {
            "pdf": pdf_path.name,
            "refs_start_page_reported": refs_start_page,
            "error": "No pages rendered or no usable references found",
            "pages_sent": 0,
            "references_found": 0,
            "references": [],
        }

    return best_result


def run(
    pdf_path: Path,
    out_dir: Path,
    extract_tables_with_deepseek: bool = EXTRACT_TABLES_WITH_DEEPSEEK,
    extract_references: bool = EXTRACT_REFERENCES,
) -> None:
    start_time = time.perf_counter()

    pdf_path = Path(pdf_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[MinerU + DeepSeek] Step 1: MinerU table crops", flush=True)

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

    _write_json(
        out_dir / "refs_start_page.json",
        {
            "pdf": pdf_path.name,
            "refs_start_page": refs_start_page,
        },
    )

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

        print("[MinerU + DeepSeek] No table crops found", flush=True)
        return

    if not extract_tables_with_deepseek:
        elapsed = time.perf_counter() - start_time

        _write_json(
            out_dir / "tables.json",
            {
                "pdf": pdf_path.name,
                "mineru_crops": len(png_paths),
                "tables_found": None,
                "tables": [],
                "note": "DeepSeek table OCR disabled. Only MinerU PNG crops were produced.",
                "refs_start_page": refs_start_page,
            },
        )

        (out_dir / "notes.md").write_text(
            "\n".join(
                [
                    "- Library: MinerU table crops only",
                    f"- PDF: {pdf_path.name}",
                    f"- mineru_crops: {len(png_paths)}",
                    f"- refs_start_page: {refs_start_page}",
                    f"- Duration: {elapsed:.2f} s",
                    "- Files: images/tables/*.png, images/tables/tables_index.json, tables.json, refs_start_page.json",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        print("[MinerU + DeepSeek] DeepSeek table OCR disabled. PNG crops only.", flush=True)
        return

    if not _deepseek_is_up():
        elapsed = time.perf_counter() - start_time

        _write_json(
            out_dir / "tables.json",
            {
                "pdf": pdf_path.name,
                "mineru_crops": len(png_paths),
                "tables_found": 0,
                "tables": [],
                "note": "DeepSeek was not reachable. Only MinerU PNG crops were produced.",
                "refs_start_page": refs_start_page,
            },
        )

        (out_dir / "notes.md").write_text(
            "\n".join(
                [
                    "- Library: MinerU + DeepSeek OCR",
                    f"- PDF: {pdf_path.name}",
                    f"- mineru_crops: {len(png_paths)}",
                    "- DeepSeek: skipped, health check failed",
                    f"- refs_start_page: {refs_start_page}",
                    f"- Duration: {elapsed:.2f} s",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        print("[MinerU + DeepSeek] DeepSeek not reachable. Skipped.", flush=True)
        return

    print("[MinerU + DeepSeek] Step 2: DeepSeek OCR on table crops", flush=True)

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
        DEEPSEEK_IMAGES_API,
        files=files,
        headers={"Expect": ""},
        timeout=(10, 60 * 60),
        proxies={"http": None, "https": None},
    )

    print("[MinerU + DeepSeek] DeepSeek status:", response.status_code, flush=True)

    response.raise_for_status()

    payload: Dict[str, Any] = response.json()

    _write_json(out_dir / "raw_deepseek_images_response.json", payload)

    if not payload.get("success"):
        raise RuntimeError("DeepSeek /ocr/images returned success=false")

    deepseek_tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []

    mineru_by_name = {
        table.get("png_name"): table
        for table in tables
        if isinstance(table, dict) and table.get("png_name")
    }

    merged_tables: List[Dict[str, Any]] = []

    for table in deepseek_tables:
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

    tables_output = {
        "pdf": pdf_path.name,
        "mineru_crops": len(png_paths),
        "tables_found": len([table for table in merged_tables if (table.get("n_rows") or 0) > 0]),
        "tables": merged_tables,
        "refs_start_page": refs_start_page,
    }

    _write_json(out_dir / "tables.json", tables_output)

    references_summary = "- References: DISABLED"

    if extract_references:
        if refs_start_page is None:
            references_summary = "- References: SKIPPED, refs_start_page not detected"

            _write_json(
                out_dir / "references.json",
                {
                    "pdf": pdf_path.name,
                    "refs_start_page": None,
                    "pages_sent": 0,
                    "references_found": 0,
                    "references": [],
                },
            )
        else:
            print("[MinerU + DeepSeek] Step 3: DeepSeek references page-by-page", flush=True)

            refs_result = _extract_references_page_by_page(
                pdf_path=pdf_path,
                refs_start_page=refs_start_page,
                out_dir=out_dir,
                dpi=180,
                max_pages=8,
                try_offsets=True,
            )

            _write_json(out_dir / "references.json", refs_result)

            references_summary = (
                f"- References: OK "
                f"start_used={refs_result.get('refs_start_page_used')} "
                f"pages_sent={refs_result.get('pages_sent')} "
                f"references_found={refs_result.get('references_found')}"
            )

    elapsed = time.perf_counter() - start_time

    (out_dir / "notes.md").write_text(
        "\n".join(
            [
                "- Library: MinerU + DeepSeek OCR",
                f"- PDF: {pdf_path.name}",
                f"- mineru_crops: {len(png_paths)}",
                f"- tables_found: {tables_output['tables_found']}",
                f"- refs_start_page_reported: {refs_start_page}",
                references_summary,
                f"- Duration: {elapsed:.2f} s",
                "- Files: images/tables/*.png, tables_index.json, tables.json, refs_start_page.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("[MinerU + DeepSeek] Done.", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MinerU crops and optional DeepSeek OCR.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--deepseek-tables", action="store_true", help="Run DeepSeek OCR on MinerU table crops.")
    parser.add_argument("--deepseek-refs", action="store_true", help="Run DeepSeek reference extraction.")

    args = parser.parse_args()

    run(
        pdf_path=Path(args.pdf),
        out_dir=Path(args.out),
        extract_tables_with_deepseek=args.deepseek_tables,
        extract_references=args.deepseek_refs,
    )