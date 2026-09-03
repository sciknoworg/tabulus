from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict

import requests

KREUZBERG_API_URL = "http://127.0.0.1:8010/extract"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_paper_nr(path: Path) -> str:
    match = re.search(r"P\d+", path.stem, re.IGNORECASE)
    if match:
        return match.group(0).upper()

    match = re.search(r"P\d+", str(path), re.IGNORECASE)
    if match:
        return match.group(0).upper()

    return path.stem


def _post_pdf(pdf_path: Path) -> Dict[str, Any]:
    with pdf_path.open("rb") as file:
        response = requests.post(
            KREUZBERG_API_URL,
            files={"files": (pdf_path.name, file, "application/pdf")},
            headers={"Expect": ""},
            timeout=(10, 60 * 60),
            proxies={"http": None, "https": None},
        )

    response.raise_for_status()

    payload = response.json()

    if isinstance(payload, list) and payload:
        payload = payload[0]

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Kreuzberg response format: {type(payload)}")

    return payload


def _extract_text_from_payload(payload: Dict[str, Any]) -> str:
    text = payload.get("content", "")

    if not isinstance(text, str):
        return ""

    return text.strip()


def _cut_references_by_page_split(text: str, start_page: int) -> str:
    pages = text.split("\n\n")

    if len(pages) < start_page:
        print("[Kreuzberg] Warning: page split unreliable. Using full text fallback.", flush=True)
        return text

    return "\n\n".join(pages[start_page - 1 :]).strip()


def run(pdf_path: Path, start_page: int, out_dir: Path | None = None) -> None:
    start_time = time.perf_counter()

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"Input file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected PDF file, got: {pdf_path.suffix}")

    if start_page < 1:
        raise ValueError("start_page must be 1-based and >= 1")

    paper_nr = extract_paper_nr(pdf_path)

    if out_dir is None:
        out_dir = pdf_path.parent / "Ref"
    else:
        out_dir = Path(out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"kreuzberg_ref_prediction_{paper_nr}"

    txt_path = out_dir / f"{base_name}.txt"
    json_path = out_dir / f"{base_name}.json"

    print(f"[Kreuzberg] Input: {pdf_path}", flush=True)
    print(f"[Kreuzberg] Paper ID: {paper_nr}", flush=True)
    print(f"[Kreuzberg] Reference start page: {start_page}", flush=True)
    print(f"[Kreuzberg] Output dir: {out_dir}", flush=True)

    payload = _post_pdf(pdf_path)
    raw_text = _extract_text_from_payload(payload)

    if not raw_text:
        raise RuntimeError("Kreuzberg returned no text content.")

    reference_text = _cut_references_by_page_split(raw_text, start_page)

    elapsed = time.perf_counter() - start_time

    txt_path.write_text(reference_text, encoding="utf-8")

    result = {
        "pdf": str(pdf_path),
        "paper_nr": paper_nr,
        "start_page": start_page,
        "output_txt": str(txt_path),
        "duration_seconds": round(elapsed, 3),
        "raw_text_length": len(raw_text),
        "reference_text_length": len(reference_text),
        "raw_response_metadata": {
            key: value for key, value in payload.items() if key != "content"
        },
    }

    _write_json(json_path, result)

    print(f"[Kreuzberg] Saved TXT: {txt_path}", flush=True)
    print(f"[Kreuzberg] Saved JSON: {json_path}", flush=True)
    print(f"[Kreuzberg] Runtime: {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract raw reference text with Kreuzberg OCR.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF.")
    parser.add_argument("--start-page", required=True, type=int, help="1-based reference start page.")
    parser.add_argument("--out", required=False, help="Output directory. Default: PDF parent / Ref.")

    args = parser.parse_args()

    run(
        pdf_path=Path(args.pdf),
        start_page=args.start_page,
        out_dir=Path(args.out) if args.out else None,
    )