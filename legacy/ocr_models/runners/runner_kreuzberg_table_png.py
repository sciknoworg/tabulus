from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import requests

KREUZBERG_API_URL = "http://127.0.0.1:8010/extract"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def rows_to_html(rows: List[List[str]]) -> str:
    if not rows:
        return "<table></table>"

    output = ["<table><thead><tr>"]

    for cell in rows[0]:
        output.append(f"<th>{html.escape(str(cell))}</th>")

    output.append("</tr></thead>")

    if len(rows) > 1:
        output.append("<tbody>")

        for row in rows[1:]:
            output.append("<tr>")

            for cell in row:
                output.append(f"<td>{html.escape(str(cell))}</td>")

            output.append("</tr>")

        output.append("</tbody>")

    output.append("</table>")
    return "".join(output)


def write_md(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows_to_html(rows), encoding="utf-8")


def extract_rows(payload: Any) -> List[List[str]]:
    if isinstance(payload, list) and payload:
        payload = payload[0]

    if not isinstance(payload, dict):
        return []

    content = payload.get("content", "")

    if not isinstance(content, str) or not content.strip():
        return []

    lines = [line.strip() for line in content.splitlines() if line.strip()]

    if not lines:
        return []

    rows: List[List[str]] = []

    header = re.split(r"\s{2,}", lines[0])
    header = [cell.strip() for cell in header if cell.strip()]

    if len(header) <= 1:
        header = lines[0].split()

    rows.append(header)

    for line in lines[1:]:
        parts = re.split(r"\s{2,}", line)
        parts = [part.strip() for part in parts if part.strip()]

        if len(parts) <= 1:
            parts = line.split()

        rows.append(parts)

    return rows


def _post_png(png_path: Path) -> Dict[str, Any] | List[Any]:
    with png_path.open("rb") as file:
        response = requests.post(
            KREUZBERG_API_URL,
            files={"files": (png_path.name, file, "image/png")},
            headers={"Expect": ""},
            timeout=(10, 60 * 60),
            proxies={"http": None, "https": None},
        )

    response.raise_for_status()
    return response.json()


def run(png_path: Path, out_dir: Path | None = None) -> None:
    png_path = Path(png_path).resolve()

    if not png_path.exists():
        raise FileNotFoundError(f"PNG not found: {png_path}")

    if png_path.suffix.lower() != ".png":
        raise ValueError(f"Expected PNG file, got: {png_path.suffix}")

    if out_dir is None:
        out_dir = png_path.parent.parent / "Kreuzberg_prediction"
    else:
        out_dir = Path(out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = png_path.stem

    csv_path = out_dir / f"{base_name}.csv"
    json_path = out_dir / f"{base_name}.json"
    md_path = out_dir / f"{base_name}.md"

    print(f"[Kreuzberg Table] Input: {png_path}", flush=True)
    print(f"[Kreuzberg Table] Output: {out_dir}", flush=True)

    payload = _post_png(png_path)

    write_json(json_path, payload)

    rows = extract_rows(payload)

    print(f"[Kreuzberg Table] Rows found: {len(rows)}", flush=True)

    write_csv(csv_path, rows)
    write_md(md_path, rows)

    print(f"[Kreuzberg Table] CSV: {csv_path}", flush=True)
    print(f"[Kreuzberg Table] JSON: {json_path}", flush=True)
    print(f"[Kreuzberg Table] MD: {md_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kreuzberg OCR on one table PNG.")
    parser.add_argument("--png", required=True, help="Path to table PNG.")
    parser.add_argument("--out", required=False, help="Output directory. Default: ../Kreuzberg_prediction")

    args = parser.parse_args()

    run(
        png_path=Path(args.png),
        out_dir=Path(args.out) if args.out else None,
    )