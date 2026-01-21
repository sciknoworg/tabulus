from __future__ import annotations

import os
import csv
from pathlib import Path

from img2table.document import PDF
from img2table.ocr import TesseractOCR


def _write_csv(path: Path, rows: list[list[str | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow([("" if c is None else str(c)) for c in r])


def run(pdf_path: Path, out_dir: Path):
    tables_dir = out_dir / "extracted_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    notes = [
        "- Library: img2table",
        f"- PDF: {pdf_path.name}",
        "- OCR: TesseractOCR(lang='eng')",
        "- Settings:",
        "  - borderless_tables=True",
        "  - min_confidence=50",
        "- Attempts:",
    ]

    print(f"[img2table] Loading PDF: {pdf_path.name}")

    # 🔧 Make Tesseract discoverable at runtime (no global PATH needed)
    os.environ["PATH"] = r"C:\Program Files\Tesseract-OCR;" + os.environ.get("PATH", "")

    # OCR engine
    ocr = TesseractOCR(
        n_threads=1,
        lang="eng",
    )

    doc = PDF(str(pdf_path))

    try:
        extracted = doc.extract_tables(
            ocr=ocr,
            borderless_tables=True,
            min_confidence=50,
        )
    except Exception as e:
        notes.append(f"  - FAILED: {type(e).__name__}: {e}")
        (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
        print(f"[img2table] FAILED: {type(e).__name__}: {e}")
        return

    total_tables = 0
    saved_tables = 0

    # extracted: {page_number: [ExtractedTable, ...]}
    for page_num, tables in (extracted or {}).items():
        total_tables += len(tables)

        for t_idx, t in enumerate(tables, start=1):
            rows = None

            # Preferred: pandas DataFrame
            if hasattr(t, "df") and t.df is not None:
                rows = [list(t.df.columns)] + t.df.values.tolist()

            # Fallback: raw content
            elif hasattr(t, "content") and t.content is not None:
                rows = t.content

            if not rows:
                continue

            out_file = tables_dir / f"page_{int(page_num):03d}_table_{t_idx:02d}.csv"
            _write_csv(out_file, rows)
            saved_tables += 1

    notes.append(f"  - extracted_tables={total_tables}")
    notes.append(f"  - saved_tables={saved_tables}")

    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(
        f"[img2table] extracted={total_tables}, "
        f"saved={saved_tables}, "
        f"out={tables_dir}"
    )
