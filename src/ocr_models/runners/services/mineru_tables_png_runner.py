from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _mineru_cmd() -> list[str]:
    if shutil.which("mineru"):
        return ["mineru"]

    py = shutil.which("python") or shutil.which("python3")
    if not py:
        raise RuntimeError("Neither `mineru` nor `python` found on PATH.")

    return [py, "-m", "mineru"]


def _run(cmd: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        shell=False,
    )

    stdout = process.stdout.decode("utf-8", errors="replace") if process.stdout else ""
    stderr = process.stderr.decode("utf-8", errors="replace") if process.stderr else ""

    return process.returncode, stdout, stderr


def _find_file(mineru_out: Path, stem: str, suffix: str) -> Optional[Path]:
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


def _resolve_img_path(
    mineru_out: Path,
    content_list_json: Path,
    img_path: str,
) -> Optional[Path]:
    path = Path(img_path)

    if path.is_file():
        return path

    candidate = mineru_out / img_path
    if candidate.is_file():
        return candidate

    base = content_list_json.parent
    candidate = base / img_path
    if candidate.is_file():
        return candidate

    filename = path.name

    hits = list(base.rglob(filename))
    if hits:
        return hits[0]

    hits = list(mineru_out.rglob(filename))
    return hits[0] if hits else None


REF_PATTERNS = [
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*literaturverzeichnis\s*$",
    r"^\s*quellen\s*$",
    r"^\s*referenzen\s*$",
]


def _text_of_item(item: dict) -> str:
    for key in ("text", "content", "raw_text", "title"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _find_refs_start_page(items: list[dict]) -> Optional[int]:
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in REF_PATTERNS]
    last_match_page = None

    for item in items:
        page_idx = item.get("page_idx")
        if not isinstance(page_idx, int):
            continue

        text = _text_of_item(item)
        if not text:
            continue

        is_heading = item.get("type") in ("title", "heading")
        is_short_heading = len(text) <= 40 and "\n" not in text

        if (is_heading or is_short_heading) and any(pattern.match(text) for pattern in patterns):
            last_match_page = page_idx + 1

    return last_match_page


def _copy_table_images_from_content_list(
    mineru_out: Path,
    content_list_json: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    crops_dir = out_dir / "images" / "tables"
    crops_dir.mkdir(parents=True, exist_ok=True)

    items = _load_content_list_items(content_list_json)
    refs_start_page = _find_refs_start_page(items)

    table_items = [item for item in items if item.get("type") == "table"]

    index_rows: List[Dict[str, Any]] = []
    saved = 0

    for table_id, table in enumerate(table_items, start=1):
        img_path = table.get("img_path")

        if not isinstance(img_path, str) or not img_path.strip():
            continue

        source = _resolve_img_path(mineru_out, content_list_json, img_path)

        if not source:
            continue

        page_idx = table.get("page_idx")
        page_nr = page_idx + 1 if isinstance(page_idx, int) else None

        dst_name = f"page_{(page_nr or 0):03d}_table_{table_id:03d}.png"
        dst = crops_dir / dst_name

        shutil.copyfile(source, dst)
        saved += 1

        in_references = bool(refs_start_page and page_nr and page_nr >= refs_start_page)

        index_rows.append(
            {
                "table_id": table_id,
                "page_nr": page_nr,
                "in_references": in_references,
                "png": str(dst.resolve()),
                "png_name": dst_name,
                "mineru_src": str(source),
                "mineru_img_path": img_path,
                "bbox": table.get("bbox"),
                "table_caption": table.get("table_caption"),
                "table_footnote": table.get("table_footnote"),
            }
        )

    index_path = crops_dir / "tables_index.json"
    index_path.write_text(
        json.dumps(
            {
                "tables_found": len(index_rows),
                "crops_saved": saved,
                "refs_start_page": refs_start_page,
                "tables": index_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "tables_found": len(index_rows),
        "crops_saved": saved,
        "refs_start_page": refs_start_page,
    }


def run(pdf_path: Path, out_dir: Path) -> None:
    start_time = time.perf_counter()

    pdf_path = Path(pdf_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mineru_out = out_dir / "mineru_out"
    mineru_out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MINERU_DEVICE_MODE"] = os.getenv("MINERU_DEVICE_MODE", "cuda")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    backend = os.getenv("MINERU_BACKEND", "pipeline")

    cmd = [
        *_mineru_cmd(),
        "-p",
        str(pdf_path),
        "-o",
        str(mineru_out),
        "-b",
        backend,
        "-m",
        "ocr",
        "-t",
        "true",
        "-f",
        "false",
    ]

    print(f"[MinerU] Running on {pdf_path.name}", flush=True)

    return_code, stdout, stderr = _run(cmd, env=env)

    (out_dir / "mineru_stdout.log").write_text(stdout or "", encoding="utf-8")
    (out_dir / "mineru_stderr.log").write_text(stderr or "", encoding="utf-8")

    if return_code != 0 or "Traceback" in stderr or "ModuleNotFoundError" in stderr:
        elapsed = time.perf_counter() - start_time

        (out_dir / "notes.md").write_text(
            "\n".join(
                [
                    "- Library: MinerU table crop extraction",
                    f"- PDF: {pdf_path.name}",
                    f"- Exit code: {return_code}",
                    "- Result: FAILED",
                    "- See: mineru_stderr.log",
                    f"- Duration: {elapsed:.2f} s",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        raise RuntimeError("MinerU failed. See mineru_stderr.log")

    content_list_json = _find_file(mineru_out, pdf_path.stem, "content_list")

    if not content_list_json:
        elapsed = time.perf_counter() - start_time

        (out_dir / "notes.md").write_text(
            "\n".join(
                [
                    "- Library: MinerU table crop extraction",
                    f"- PDF: {pdf_path.name}",
                    "- Result: OK, but *_content_list.json was not found",
                    f"- Duration: {elapsed:.2f} s",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        print("[MinerU] Done, but content_list.json not found", flush=True)
        return

    stats = _copy_table_images_from_content_list(mineru_out, content_list_json, out_dir)

    elapsed = time.perf_counter() - start_time

    (out_dir / "notes.md").write_text(
        "\n".join(
            [
                "- Library: MinerU table crop extraction",
                f"- PDF: {pdf_path.name}",
                f"- content_list.json: {content_list_json.name}",
                f"- tables_found: {stats['tables_found']}",
                f"- crops_saved: {stats['crops_saved']}",
                f"- refs_start_page: {stats.get('refs_start_page')}",
                f"- Duration: {elapsed:.2f} s",
                "- Output: images/tables/*.png + images/tables/tables_index.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[MinerU] Done. tables_found={stats['tables_found']} "
        f"crops_saved={stats['crops_saved']} "
        f"refs_start_page={stats.get('refs_start_page')}",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MinerU table crop extraction.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF.")
    parser.add_argument("--out", required=True, help="Output directory.")

    args = parser.parse_args()

    run(Path(args.pdf), Path(args.out))