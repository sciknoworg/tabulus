from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# -------------------------
# subprocess (Windows-safe)
# -------------------------


def _mineru_cmd() -> list[str]:
    if shutil.which("mineru"):
        return ["mineru"]
    py = shutil.which("python") or shutil.which("python3")
    if not py:
        raise RuntimeError("Neither `mineru` nor `python` found on PATH.")
    return [py, "-m", "mineru"]


def _run(cmd: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        shell=False,
    )
    stdout = p.stdout.decode("utf-8", errors="replace") if p.stdout else ""
    stderr = p.stderr.decode("utf-8", errors="replace") if p.stderr else ""
    return p.returncode, stdout, stderr


# -------------------------
# MinerU output discovery
# -------------------------


def _find_file(mineru_out: Path, stem: str, suffix: str) -> Optional[Path]:
    """
    Find files like <stem>_content_list.json anywhere under mineru_out.
    """
    direct = mineru_out / f"{stem}_{suffix}.json"
    if direct.exists():
        return direct
    matches = list(mineru_out.rglob(f"{stem}_{suffix}.json"))
    return matches[0] if matches else None


def _load_content_list_items(content_list_json: Path) -> List[dict]:
    data: Any = json.loads(content_list_json.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        items = data.get("content_list") or data.get("items") or data.get("data") or []
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]

    return []


def _resolve_img_path(mineru_out: Path, content_list_json: Path, img_path: str) -> Optional[Path]:
    """
    img_path can be:
      - absolute
      - relative to mineru_out
      - relative to the folder that contains content_list_json (common)
    We'll try all robust options + a filename search fallback.
    """
    p = Path(img_path)

    # absolute
    if p.is_file():
        return p

    # relative to mineru_out root
    p2 = mineru_out / img_path
    if p2.is_file():
        return p2

    # relative to content_list_json folder
    base = content_list_json.parent
    p3 = base / img_path
    if p3.is_file():
        return p3

    # common MinerU layout: mineru_out/<stem>/ocr/images/<hash>.png
    # if img_path is like "ocr/images/<hash>.png" then base/<img_path> works.
    # if img_path is just "<hash>.png", search by filename.
    fname = p.name
    hits = list(base.rglob(fname))
    if hits:
        return hits[0]

    hits2 = list(mineru_out.rglob(fname))
    return hits2[0] if hits2 else None


# -------------------------
# NEW: references detection
# -------------------------

_REF_PATTERNS = [
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*literaturverzeichnis\s*$",
    r"^\s*quellen\s*$",
    r"^\s*referenzen\s*$",
]


def _text_of_item(it: dict) -> str:
    # Try common MinerU keys for text-like content
    for k in ("text", "content", "raw_text", "title"):
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _find_refs_start_page(items: list[dict]) -> Optional[int]:
    """
    Returns 1-based page number where references start, or None.
    Picks the LAST matching heading in the document.
    """
    pats = [re.compile(p, re.IGNORECASE) for p in _REF_PATTERNS]

    last_match_page = None

    for it in items:
        page_idx = it.get("page_idx")
        if not isinstance(page_idx, int):
            continue

        t = _text_of_item(it)
        if not t:
            continue

        is_headingish = (it.get("type") in ("title", "heading")) or (
            len(t) <= 40 and "\n" not in t
        )

        if is_headingish and any(p.match(t) for p in pats):
            last_match_page = page_idx + 1

    return last_match_page

def _copy_table_images_from_content_list(mineru_out: Path, content_list_json: Path, out_dir: Path) -> Dict[str, Any]:
    crops_dir = out_dir / "images" / "tables"
    crops_dir.mkdir(parents=True, exist_ok=True)

    items = _load_content_list_items(content_list_json)
    refs_start_page = _find_refs_start_page(items)  # NEW

    table_items = [it for it in items if it.get("type") == "table"]

    index_rows: List[Dict[str, Any]] = []
    saved = 0

    for i, t in enumerate(table_items, start=1):
        img_path = t.get("img_path")
        if not isinstance(img_path, str) or not img_path.strip():
            continue

        src = _resolve_img_path(mineru_out, content_list_json, img_path)
        if not src:
            continue

        page_idx = t.get("page_idx")
        page_nr = (page_idx + 1) if isinstance(page_idx, int) else None

        dst_name = f"page_{(page_nr or 0):03d}_table_{i:03d}.png"
        dst = crops_dir / dst_name
        shutil.copyfile(src, dst)
        saved += 1

        # NEW: mark tables that are in/after references section
        in_references = bool(refs_start_page and page_nr and page_nr >= refs_start_page)

        index_rows.append(
            {
                "table_id": i,
                "page_nr": page_nr,
                "in_references": in_references,  # NEW
                "png": str(dst.resolve()),
                "png_name": dst_name,
                "mineru_src": str(src),
                "mineru_img_path": img_path,
                "bbox": t.get("bbox"),
                "table_caption": t.get("table_caption"),
                "table_footnote": t.get("table_footnote"),
            }
        )

    (crops_dir / "tables_index.json").write_text(
        json.dumps(
            {
                "tables_found": len(index_rows),
                "crops_saved": saved,
                "refs_start_page": refs_start_page,  # NEW
                "tables": index_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"tables_found": len(index_rows), "crops_saved": saved, "refs_start_page": refs_start_page}  # NEW


# -------------------------
# Runner
# -------------------------


def run(pdf_path: Path, out_dir: Path):
    t0 = time.perf_counter()

    pdf_path = Path(pdf_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mineru_out = out_dir / "mineru_out"
    mineru_out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MINERU_DEVICE_MODE"] = "cuda"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        *_mineru_cmd(),
        "-p", str(pdf_path),
        "-o", str(mineru_out),
        "-b", "Tabulus",
        "--device", "cuda",
        "-m", "ocr",
        "-t", "true",
        "-f", "false",
    ]

    print(f"[MinerU-TablesPNG] Running on {pdf_path.name} ...", flush=True)
    rc, stdout, stderr = _run(cmd, env=env)

    (out_dir / "mineru_stdout.log").write_text(stdout or "", encoding="utf-8")
    (out_dir / "mineru_stderr.log").write_text(stderr or "", encoding="utf-8")

    if rc != 0 or "Traceback" in stderr or "ModuleNotFoundError" in stderr:
        elapsed = time.perf_counter() - t0
        (out_dir / "notes.md").write_text(
            "\n".join(
                [
                    "- Library: MinerU (tables PNG only via content_list img_path)",
                    f"- PDF: {pdf_path.name}",
                    f"- Exit code: {rc}",
                    "- Result: FAILED (see mineru_stderr.log)",
                    f"- Duration: {elapsed:.2f} s",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print("[MinerU-TablesPNG] FAILED. See mineru_stderr.log", flush=True)
        return

    stem = pdf_path.stem
    content_list_json = _find_file(mineru_out, stem, "content_list")

    if not content_list_json:
        elapsed = time.perf_counter() - t0
        (out_dir / "notes.md").write_text(
            "\n".join(
                [
                    "- Library: MinerU (tables PNG only via content_list img_path)",
                    f"- PDF: {pdf_path.name}",
                    "- Result: OK but *_content_list.json not found",
                    f"- Duration: {elapsed:.2f} s",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print("[MinerU-TablesPNG] Done but *_content_list.json not found.", flush=True)
        return

    stats = _copy_table_images_from_content_list(mineru_out, content_list_json, out_dir)

    elapsed = time.perf_counter() - t0
    (out_dir / "notes.md").write_text(
        "\n".join(
            [
                "- Library: MinerU (tables PNG only via content_list img_path)",
                f"- PDF: {pdf_path.name}",
                f"- content_list.json: {content_list_json.name}",
                f"- tables_found: {stats['tables_found']}",
                f"- crops_saved: {stats['crops_saved']}",
                f"- refs_start_page: {stats.get('refs_start_page')}",  # NEW (info only)
                f"- Duration: {elapsed:.2f} s",
                "- Output: images/tables/*.png + images/tables/tables_index.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[MinerU-TablesPNG] Done. tables_found={stats['tables_found']} crops_saved={stats['crops_saved']} refs_start_page={stats.get('refs_start_page')}",
        flush=True,
    )
