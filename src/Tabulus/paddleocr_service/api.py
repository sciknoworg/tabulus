from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import tempfile
import time
import json
import re
from html.parser import HTMLParser

import fitz  # pymupdf
from PIL import Image
from paddleocr import PaddleOCRVL

app = FastAPI()


def _clean_text_refs(text: str) -> str:
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_refs(text: str):
    text = _clean_text_refs(text)

    text = re.sub(
        r'(?i)(doi\s*:\s*10\.[^\n]+)\n\s*([0-9A-Za-z.\-_/]+)',
        lambda m: m.group(1).rstrip() + m.group(2).lstrip(),
        text
    )

    text = re.sub(
        r'(?i)(10\.\S+?)\n\s*([0-9A-Za-z.\-_/]+)',
        r'\1\2',
        text
    )

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    refs = []
    current = []
    started = False

    year_pattern = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
    numbered_start = re.compile(r"^\s*\d+[\.\)]\s+")
    doi_start = re.compile(r"^(doi:|https?://doi\.org/)", re.I)
    stop_garbage = re.compile(
        r"(?i)^(conflict of interest|publisher[’']?s note|copyright|open access)\b"
    )

    def is_reference_heading(line: str) -> bool:
        s = re.sub(r"\s+", " ", line).strip().lower()
        return s in {"references", "bibliography", "literature cited"}

    def is_safe_continuation(line: str) -> bool:
        s = line.strip()
        if not s:
            return False

        if doi_start.match(s):
            return True

        if re.match(r"^\(?accessed\b", s, re.I):
            return True

        if re.match(r"^(available at:|retrieved from:)", s, re.I):
            return True

        if re.match(r"^[a-zà-öø-ÿ\(\[\{]", s):
            return True

        if re.match(r"^(vol\.?|no\.?|pp\.?|pages?)\b", s, re.I):
            return True

        if re.match(r"^\d+\s*[-–]\s*\d+$", s):
            return True

        if len(s) > 1 and s[0].islower():
            return True

        return False

    def looks_like_new_reference(line: str) -> bool:
        s = line.strip()
        if not s:
            return False

        if numbered_start.match(s) and year_pattern.search(s):
            return True

        if s[0].isupper() and year_pattern.search(s):
            return True

        return False

    def finalize():
        nonlocal current, refs
        if not current:
            return

        ref = " ".join(current)
        ref = re.sub(r"\s+", " ", ref).strip()

        ref = re.split(
            r"(?i)\b(conflict of interest|publisher[’']?s note|copyright|open access)\b",
            ref
        )[0].strip()

        if ref:
            refs.append(ref)

        current = []

    for line in lines:
        if not started:
            if is_reference_heading(line):
                started = True
                continue

            if looks_like_new_reference(line):
                started = True
            else:
                continue

        if stop_garbage.match(line):
            break

        if not current:
            current = [line]
            continue

        if is_safe_continuation(line):
            current.append(line)
            continue

        if looks_like_new_reference(line):
            finalize()
            current = [line]
            continue

        current.append(line)

    finalize()

    return [{"nr": i, "ref": r} for i, r in enumerate(refs, 1)]


print("[api] loading PaddleOCRVL...", flush=True)
pipeline = PaddleOCRVL()
print("[api] PaddleOCRVL loaded ✅", flush=True)


def render_pdf_page_to_image(pdf_path: str, page_index: int, dpi: int = 200) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _get_markdown_from_res(res) -> str:
    md = None
    if hasattr(res, "markdown"):
        md = res.markdown
    if isinstance(md, str) and md.strip():
        return md
    if isinstance(md, dict):
        for k in ("markdown_texts", "markdown", "text", "md", "output"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(md, ensure_ascii=False, indent=2)
    s = str(res)
    return s if s.strip() else ""


def _extract_html_tables(text: str) -> list[str]:
    return re.findall(r"(?is)<table\b.*?</table>", text)


class _SimpleTableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_tr = False
        self.in_td = False
        self.current_cell = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self.in_tr = True
            self.current_row = []
        elif tag in ("td", "th") and self.in_tr:
            self.in_td = True
            self.current_cell = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self.in_td:
            self.in_td = False
            cell_text = "".join(self.current_cell).strip()
            cell_text = re.sub(r"\s+", " ", cell_text)
            self.current_row.append(cell_text)
            self.current_cell = []
        elif tag == "tr" and self.in_tr:
            self.in_tr = False
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []

    def handle_data(self, data):
        if self.in_td and data:
            self.current_cell.append(data)


def _html_table_to_rows(html: str) -> list[list[str]]:
    parser = _SimpleTableHTMLParser()
    parser.feed(html)
    rows = parser.rows
    max_cols = max((len(r) for r in rows), default=0)
    return [r + [""] * (max_cols - len(r)) for r in rows]


def _extract_markdown_tables(text: str) -> list[str]:
    lines = text.splitlines()
    out: list[str] = []

    def is_table_line(s: str) -> bool:
        s = s.strip()
        return "|" in s and len(s) >= 3

    def is_sep(s: str) -> bool:
        s = s.strip().replace(" ", "")
        return ("---" in s) and ("|" in s)

    i = 0
    while i < len(lines):
        if is_table_line(lines[i]):
            buf = [lines[i]]
            saw_sep = is_sep(lines[i])
            j = i + 1
            while j < len(lines) and is_table_line(lines[j]):
                buf.append(lines[j])
                if is_sep(lines[j]):
                    saw_sep = True
                j += 1
            if saw_sep and len(buf) >= 2:
                out.append("\n".join(buf).strip() + "\n")
            i = j
        else:
            i += 1
    return out


def _markdown_table_to_rows(md_table: str) -> list[list[str]]:
    rows = [r.strip() for r in md_table.splitlines() if r.strip()]
    if len(rows) < 2:
        return []

    sep_idx = None
    for i, r in enumerate(rows):
        if "|" in r and "---" in r:
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return []

    def split_row(r: str) -> list[str]:
        parts = [c.strip() for c in r.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return parts

    header = split_row(rows[0])
    body = [split_row(r) for r in rows[sep_idx + 1:]]

    max_cols = max([len(header)] + [len(r) for r in body], default=len(header))
    header = header + [""] * (max_cols - len(header))
    body = [r + [""] * (max_cols - len(r)) for r in body]

    return [header] + body


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...)):
    t_start = time.time()
    print(f"[api] /ocr/pdf START file={file.filename}", flush=True)

    pdf_bytes = await file.read()
    print(f"[api] upload received: {len(pdf_bytes)/1024/1024:.2f} MB", flush=True)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        pdf_path = f.name

    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    doc.close()
    print(f"[api] pages detected: {n_pages}", flush=True)

    run_stem = Path(file.filename).stem if file.filename else "upload"
    run_dir = Path("/tmp/ocr_runs") / run_stem
    run_dir.mkdir(parents=True, exist_ok=True)

    combined_parts: list[str] = []
    tables_out: list[dict] = []
    table_id = 0

    for i in range(n_pages):
        page_nr = i + 1
        page_dir = run_dir / f"page_{page_nr:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        img = render_pdf_page_to_image(pdf_path, i, dpi=200)
        img_path = page_dir / "page.png"
        img.save(img_path)

        print(f"[api] page {page_nr}/{n_pages}: predict START", flush=True)
        t_page = time.time()

        output = pipeline.predict(str(img_path))

        print(f"[api] page {page_nr}/{n_pages}: predict DONE ({time.time()-t_page:.1f}s)", flush=True)

        try:
            res0 = next(iter(output))
        except Exception:
            res0 = output

        md_text = _get_markdown_from_res(res0)
        if not md_text.strip():
            md_text = f"<!-- PAGE {page_nr:03d} -->\n(OCR returned empty markdown)\n"

        combined_parts.append(f"\n\n<!-- PAGE {page_nr:03d} START -->\n")
        combined_parts.append(md_text.rstrip() + "\n")
        combined_parts.append(f"<!-- PAGE {page_nr:03d} END -->\n")

        html_tables = _extract_html_tables(md_text)
        for html in html_tables:
            table_id += 1
            rows = _html_table_to_rows(html)
            tables_out.append(
                {
                    "table_id": table_id,
                    "page_nr": page_nr,
                    "n_rows": len(rows),
                    "n_cols": max((len(r) for r in rows), default=0),
                    "rows": rows,
                }
            )

        print(f"[api] page {page_nr}/{n_pages}: tables_found={len(html_tables)}", flush=True)

    result_mmd = "".join(combined_parts)
    print(f"[api] /ocr/pdf DONE total_time={time.time()-t_start:.1f}s tables_total={len(tables_out)}", flush=True)

    return {
        "success": True,
        "file": file.filename,
        "pages": n_pages,
        "tables_found": len(tables_out),
        "result_mmd": result_mmd,
        "tables": tables_out,
    }


@app.post("/ocr/images")
async def ocr_images(files: List[UploadFile] = File(...)):
    t_start = time.time()
    print(f"[api] /ocr/images START n_files={len(files)}", flush=True)

    run_dir = Path("/tmp/ocr_image_runs") / str(int(t_start))
    run_dir.mkdir(parents=True, exist_ok=True)

    tables_out: list[dict] = []
    table_id = 0

    for idx, uf in enumerate(files, start=1):
        name = uf.filename or f"image_{idx}.png"
        img_bytes = await uf.read()

        img_path = run_dir / name
        img_path.write_bytes(img_bytes)

        print(f"[api] /ocr/images {idx}/{len(files)} predict START name={name}", flush=True)
        t_one = time.time()

        try:
            output = pipeline.predict(str(img_path))
            try:
                res0 = next(iter(output))
            except Exception:
                res0 = output

            md_text = _get_markdown_from_res(res0).strip()
        except Exception as e:
            print(f"[api] /ocr/images ERROR name={name}: {type(e).__name__}: {e}", flush=True)
            tables_out.append(
                {
                    "table_id": None,
                    "source_file": name,
                    "error": f"{type(e).__name__}: {e}",
                    "n_rows": 0,
                    "n_cols": 0,
                    "rows": [],
                }
            )
            continue

        print(f"[api] /ocr/images {idx}/{len(files)} predict DONE ({time.time()-t_one:.1f}s)", flush=True)

        html_tables = _extract_html_tables(md_text)

        if html_tables:
            for html in html_tables:
                table_id += 1
                rows = _html_table_to_rows(html)
                tables_out.append(
                    {
                        "table_id": table_id,
                        "source_file": name,
                        "n_rows": len(rows),
                        "n_cols": max((len(r) for r in rows), default=0),
                        "rows": rows,
                        "source": "html",
                    }
                )
        else:
            md_tables = _extract_markdown_tables(md_text)
            if md_tables:
                for mt in md_tables:
                    rows = _markdown_table_to_rows(mt)
                    if not rows:
                        continue
                    table_id += 1
                    tables_out.append(
                        {
                            "table_id": table_id,
                            "source_file": name,
                            "n_rows": len(rows),
                            "n_cols": max((len(r) for r in rows), default=0),
                            "rows": rows,
                            "source": "markdown",
                        }
                    )
            else:
                table_id += 1
                tables_out.append(
                    {
                        "table_id": table_id,
                        "source_file": name,
                        "n_rows": 0,
                        "n_cols": 0,
                        "rows": [],
                        "source": "empty",
                        "raw": md_text[:2000],
                    }
                )

    print(f"[api] /ocr/images DONE total_time={time.time()-t_start:.1f}s tables_total={len(tables_out)}", flush=True)

    return {
        "success": True,
        "files": [f.filename for f in files],
        "tables_found": len([t for t in tables_out if (t.get("n_rows") or 0) > 0]),
        "tables": tables_out,
    }


@app.post("/ocr/references")
async def ocr_references(
    file: UploadFile = File(...),
    ref_start_page_nr: int = Form(...)
):
    t_start = time.time()
    print(f"[api] /ocr/references START page={ref_start_page_nr}", flush=True)

    pdf_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        pdf_path = f.name

    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    doc.close()

    all_text = []

    for page_nr in range(ref_start_page_nr, n_pages + 1):
        img = render_pdf_page_to_image(pdf_path, page_nr - 1, dpi=200)

        tmp_img = Path(tempfile.gettempdir()) / f"ref_page_{page_nr}.png"
        img.save(tmp_img)

        output = pipeline.predict(str(tmp_img))

        try:
            res0 = next(iter(output))
        except Exception:
            res0 = output

        md_text = _get_markdown_from_res(res0)

        print(f"\n--- PAGE {page_nr} OCR ---", flush=True)
        print(md_text[:800], flush=True)

        all_text.append(md_text)

    combined = "\n".join(all_text)

    print("\n=== RAW OCR TEXT ===", flush=True)
    print(combined[:3000], flush=True)
    print("=== END RAW OCR TEXT ===\n", flush=True)

    refs = _extract_refs(combined)

    print(f"[api] DONE refs={len(refs)} time={time.time()-t_start:.1f}s", flush=True)

    return {
        "success": True,
        "raw_text": combined,
        "references": refs
    }