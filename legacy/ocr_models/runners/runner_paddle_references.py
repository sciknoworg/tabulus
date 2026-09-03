from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import requests

PADDLE_BASE_URL = "http://127.0.0.1:8002"
PADDLE_REFERENCES_API = f"{PADDLE_BASE_URL}/ocr/references"
PADDLE_HEALTH_URL = f"{PADDLE_BASE_URL}/health"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_health() -> None:
    response = requests.get(PADDLE_HEALTH_URL, timeout=5)
    response.raise_for_status()

    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError("PaddleOCR-VL health check failed")


def _post_pdf(pdf_path: Path, ref_start_page_nr: int) -> Dict[str, Any]:
    with pdf_path.open("rb") as file:
        response = requests.post(
            PADDLE_REFERENCES_API,
            files={"file": (pdf_path.name, file, "application/pdf")},
            data={"ref_start_page_nr": str(ref_start_page_nr)},
            headers={"Expect": ""},
            timeout=(10, 60 * 60),
            proxies={"http": None, "https": None},
        )

    response.raise_for_status()

    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"PaddleOCR-VL /ocr/references returned success=false: {payload}")

    return payload


def run(pdf_path: Path, ref_start_page_nr: int, out_dir: Path | None = None) -> None:
    start_time = time.perf_counter()

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"Input file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.suffix}")

    if out_dir is None:
        out_dir = pdf_path.parent / "Ref"
    else:
        out_dir = Path(out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem

    json_path = out_dir / f"paddle_ref_prediction_{stem}.json"
    raw_txt_path = out_dir / f"paddle_ref_raw_{stem}.txt"

    print("[PaddleOCR-VL Refs] Step 1: health check", flush=True)
    _check_health()

    print("[PaddleOCR-VL Refs] Step 2: send PDF to references API", flush=True)
    payload = _post_pdf(pdf_path, ref_start_page_nr)

    elapsed = time.perf_counter() - start_time

    raw_text = payload.get("raw_text", "")
    references = payload.get("references", [])

    raw_txt_path.write_text(raw_text, encoding="utf-8")

    result = {
        "input_file": str(pdf_path),
        "output_dir": str(out_dir),
        "runtime_seconds": round(elapsed, 3),
        "ref_start_page_nr": ref_start_page_nr,
        "references_found": len(references),
        "references": references,
        "raw_text_file": str(raw_txt_path),
        "raw_text": raw_text,
        "raw_response": payload,
    }

    _write_json(json_path, result)

    print(f"[PaddleOCR-VL Refs] Runtime: {elapsed:.2f}s", flush=True)
    print(f"[PaddleOCR-VL Refs] JSON: {json_path}", flush=True)
    print(f"[PaddleOCR-VL Refs] Raw text: {raw_txt_path}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PaddleOCR-VL reference extraction.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF.")
    parser.add_argument("--ref-start-page", required=True, type=int, help="1-based references start page.")
    parser.add_argument("--out", required=False, help="Output directory. Default: PDF parent / Ref.")

    args = parser.parse_args()

    run(
        pdf_path=Path(args.pdf),
        ref_start_page_nr=args.ref_start_page,
        out_dir=Path(args.out) if args.out else None,
    )