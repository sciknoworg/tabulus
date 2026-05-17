# ============================
# api.py  (FastAPI DeepSeek OCR)
# ============================
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pathlib import Path
import time, re, io, tempfile, asyncio, json
from typing import List, Optional

import fitz
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from html.parser import HTMLParser

app = FastAPI()

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"

PROMPT_TABLE = (
    "<image>\n<|grounding|>"
    "Convert ONLY the table in the image to markdown. Do not add explanations."
)

# ✅ IMPORTANT: explicit NONE escape hatch
PROMPT_REFERENCES = (
    "<image>\n"
    "Task: Extract ONLY bibliography/references entries from this page.\n"
    "Rules:\n"
    "- If this page does NOT contain a bibliography/references list, output exactly: NONE\n"
    "- Do NOT include any other text (no headings, no conclusions, no style guides).\n"
    "- Keep the exact order as on the page.\n"
    "- Output plain text only.\n"
)
PROMPT_REFERENCES2 = (
    "<image>\n<|grounding|>"
    "Extract all bibliography/reference entries from this page. "
    "Return only the reference text in reading order."
)

# Globals populated after startup
tokenizer: Optional[AutoTokenizer] = None
model: Optional[torch.nn.Module] = None
model_error: Optional[str] = None

_model_init_lock = asyncio.Lock()   # protects model init
_gpu_lock = asyncio.Lock()          # serializes inference on 1 GPU

def _render_pdf_page_to_png_bytes(pdf_path: str, page_index: int, dpi: int = 200) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    doc.close()

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return buf.getvalue()


def _decode_and_resize(img_bytes: bytes, max_side: int = 2600) -> Image.Image:
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    w, h = pil.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        pil = pil.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS
        )

    return pil

def _coerce_text(obj):
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        for k in ("text", "result", "markdown", "md", "output", "response", "content"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if isinstance(obj, (list, tuple)):
        parts = []
        for item in obj:
            t = _coerce_text(item)
            if t.strip():
                parts.append(t)
        return "\n".join(parts).strip()

    for attr in ("text", "result", "markdown", "output", "response", "content"):
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            if isinstance(v, str) and v.strip():
                return v

    try:
        s = str(obj)
        return s if s.strip() else repr(obj)
    except Exception:
        try:
            return repr(obj)
        except Exception:
            return ""
# -------------------------
# Model loading (background)

# -------------------------
@app.on_event("startup")
async def startup():
    asyncio.create_task(_load_model())
    print("[api] startup: model loading scheduled ✅", flush=True)


async def _load_model():
    global tokenizer, model, model_error
    async with _model_init_lock:
        if model is not None and tokenizer is not None:
            return

        try:
            print("[api] loading tokenizer...", flush=True)
            tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

            print("[api] loading model...", flush=True)
            m = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2",
            ).eval().to("cuda")

            # ---- generation guard: less aggressive for long text ----
            orig_generate = m.generate

            def _generate_guard(*args, **kwargs):
                # hard cap to prevent runaway
                hard_cap = 8192  # ✅ lower default; references pages can be long but avoid runaway
                kwargs["max_new_tokens"] = min(int(kwargs.get("max_new_tokens", hard_cap)), hard_cap)

                # modest repetition penalty
                kwargs.setdefault("repetition_penalty", 1.08)

                # ✅ reduce the expensive constraint (30 was too heavy for long text)
                nrs = kwargs.get("no_repeat_ngram_size", None)
                if nrs is None:
                    kwargs["no_repeat_ngram_size"] = 8
                else:
                    kwargs["no_repeat_ngram_size"] = min(int(nrs), 12)

                return orig_generate(*args, **kwargs)

            m.generate = _generate_guard

            tokenizer = tok
            model = m
            model_error = None
            print("[api] model loaded ✅", flush=True)

        except Exception as e:
            tokenizer = None
            model = None
            model_error = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
            print("[api] model load FAILED ❌", flush=True)
            print(model_error, flush=True)


# -------------------------
# Helpers
# -------------------------
def _coerce_text(obj):
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ("markdown", "result", "text", "md", "output"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(obj, ensure_ascii=False, indent=2)
    try:
        return str(obj)
    except Exception:
        return ""


def _extract_html_tables(text: str) -> list[str]:
    """
    DeepSeek often emits HTML <table> ... </table>.
    Sometimes output is TRUNCATED (no closing </table>).
    We handle both:
      1) full closed tables
      2) a trailing <table ...> that runs to end of text
    """
    full = re.findall(r"(?is)<table\b.*?</table>", text)
    if full:
        return full
    m = re.search(r"(?is)(<table\b.*)$", text)
    if m:
        return [m.group(1)]
    return []


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
            self.current_row.append(re.sub(r"\s+", " ", cell_text))
            self.current_cell = []
        elif tag == "tr" and self.in_tr:
            self.in_tr = False
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []

    def handle_data(self, data):
        if self.in_td and data:
            self.current_cell.append(data)

    def close(self):
        # finalize partial cell/row if HTML is truncated
        if self.in_td:
            cell_text = "".join(self.current_cell).strip()
            if cell_text:
                self.current_row.append(re.sub(r"\s+", " ", cell_text))
            self.in_td = False
            self.current_cell = []

        if self.in_tr and self.current_row:
            self.rows.append(self.current_row)
        self.in_tr = False
        self.current_row = []
        super().close()


def _html_table_to_rows(html: str) -> list[list[str]]:
    parser = _SimpleTableHTMLParser()
    parser.feed(html)
    parser.close()
    rows = parser.rows
    max_cols = max((len(r) for r in rows), default=0)
    return [r + [""] * (max_cols - len(r)) for r in rows]


def _extract_markdown_tables(text: str) -> list[str]:
    lines = text.splitlines()
    out = []

    def is_table_line(s):
        return "|" in s and len(s.strip()) > 3

    def is_sep(s):
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

    def split_row(r):
        parts = [c.strip() for c in r.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return parts

    header = split_row(rows[0])
    body = [split_row(r) for r in rows[sep_idx + 1:]]

    max_cols = max([len(header)] + [len(r) for r in body], default=len(header))
    header += [""] * (max_cols - len(header))
    body = [r + [""] * (max_cols - len(r)) for r in body]

    return [header] + body


# -------------------------
# References helpers
# -------------------------
def _clean_ref_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)  # de-hyphenate
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip()


_REF_SPLIT_PATTERNS = [
    re.compile(r"^\s*\[\s*\d+\s*\]\s+"),            # [1] ...
    re.compile(r"^\s*\d+\.\s+"),                    # 1. ...
    re.compile(r"^\s*\d+\)\s+"),                    # 1) ...
    re.compile(r"^\s*\(\s*\d+\s*\)\s+"),            # (1) ...
    re.compile(r"^\s*\[[A-Za-z][^\]]{1,40}\]\s+"),  # [Smith2020] ...
]


def _looks_like_new_reference(line: str) -> bool:
    if not line or not line.strip():
        return False
    return any(p.match(line) for p in _REF_SPLIT_PATTERNS)


def _split_references(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    has_markers = any(_looks_like_new_reference(ln) for ln in lines)

    if has_markers:
        refs: list[str] = []
        cur: list[str] = []

        for ln in lines:
            if not ln:
                if cur:
                    cur.append("")
                continue

            if _looks_like_new_reference(ln) and cur:
                entry = "\n".join(cur).strip()
                if entry:
                    refs.append(entry)
                cur = [ln]
            else:
                cur.append(ln)

        last = "\n".join(cur).strip()
        if last:
            refs.append(last)

        return [re.sub(r"\n{3,}", "\n\n", r).strip() for r in refs if r.strip()]

    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _looks_like_bibliography(text: str) -> bool:
    """
    Heuristic to avoid garbage (style guides, conclusions, etc.).
    """
    t = (text or "").strip()
    if not t:
        return False
    if t.strip().upper() == "NONE":
        return False

    tl = t.lower()
    bad_hits = sum(x in tl for x in ["times new roman", "calibri", "spacing", "bullet points", "conclusion"])
    if bad_hits >= 2:
        return False

    hits = 0
    hits += 1 if re.search(r"\b(19|20)\d{2}\b", t) else 0
    hits += 1 if ("doi" in tl or "doi.org" in tl) else 0
    hits += 1 if ("et al" in tl) else 0
    hits += 1 if re.search(r"\bvol\.\b|\bno\.\b|\bpp\.\b", tl) else 0
    hits += 1 if re.search(r"https?://", tl) else 0

    return hits >= 2


def _decode_and_resize(img_bytes: bytes, max_side: int = 2600) -> Image.Image:
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = pil.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return pil


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_ready": model is not None,
        "model_error": None if model_error is None else model_error.splitlines()[-1],
    }


@app.post("/ocr/images")
async def ocr_images(files: List[UploadFile] = File(...)):
    """
    Input: PNG/JPG table crops
    Output: tables as JSON rows (list[list[str]])
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not ready yet. Check /health.")

    t_start = time.time()
    print(f"[api] /ocr/images START n_files={len(files)}", flush=True)

    run_dir = Path("/tmp/ocr_runs") / f"images_{int(t_start)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tables_out: list[dict] = []
    table_id = 0

    for idx, uf in enumerate(files, start=1):
        name = uf.filename or f"image_{idx}.png"
        img_bytes = await uf.read()

        if not img_bytes:
            tables_out.append({"table_id": None, "source_file": name, "error": "Empty upload", "rows": [], "n_rows": 0, "n_cols": 0})
            continue

        try:
            pil = _decode_and_resize(img_bytes, max_side=2600)
        except Exception as e:
            tables_out.append({"table_id": None, "source_file": name, "error": f"PIL decode failed: {e}", "rows": [], "n_rows": 0, "n_cols": 0})
            continue

        img_path = run_dir / name
        pil.save(img_path, format="PNG", optimize=True)

        print(f"[api] infer table {name}", flush=True)

        try:
            async with _gpu_lock:
                with torch.inference_mode():
                    infer_res = model.infer(
                        tokenizer,
                        prompt=PROMPT_TABLE,
                        image_file=str(img_path),
                        output_path=str(run_dir),
                        base_size=1536,
                        image_size=1024,
                        crop_mode=False,
                        save_results=False,
                        eval_mode=True,
                        test_compress=False,
                    )
        except Exception as e:
            tables_out.append({"table_id": None, "source_file": name, "error": f"{type(e).__name__}: {e}", "rows": [], "n_rows": 0, "n_cols": 0})
            continue

        md = (_coerce_text(infer_res) or "").strip()

        # 1) HTML tables
        html_tables = _extract_html_tables(md)
        if html_tables:
            for html in html_tables:
                table_id += 1
                rows = _html_table_to_rows(html)
                tables_out.append(
                    {"table_id": table_id, "source_file": name, "n_rows": len(rows), "n_cols": max((len(r) for r in rows), default=0), "rows": rows, "source": "html"}
                )
            continue

        # 2) Markdown tables fallback
        md_tables = _extract_markdown_tables(md)
        if md_tables:
            any_parsed = False
            for mt in md_tables:
                rows = _markdown_table_to_rows(mt)
                if not rows:
                    continue
                any_parsed = True
                table_id += 1
                tables_out.append(
                    {"table_id": table_id, "source_file": name, "n_rows": len(rows), "n_cols": max((len(r) for r in rows), default=0), "rows": rows, "source": "markdown"}
                )
            if any_parsed:
                continue

        # 3) Nothing parseable; keep raw snippet
        table_id += 1
        tables_out.append({"table_id": table_id, "source_file": name, "n_rows": 0, "n_cols": 0, "rows": [], "source": "empty", "raw": md[:2000]})

    print(f"[api] /ocr/images DONE total_time={time.time()-t_start:.1f}s tables_total={len(tables_out)}", flush=True)

    return {
        "success": True,
        "files": [f.filename for f in files],
        "tables_found": len([t for t in tables_out if (t.get("n_rows") or 0) > 0]),
        "tables": tables_out,
    }


@app.post("/ocr/references")
async def ocr_references(files: List[UploadFile] = File(...)):
    """
    Input: PNG/JPG pages (preferably sent page-by-page)
    Output:
      - If page is not references: references_found=0, references=["NONE"], is_references=False
      - If references: split entries best-effort
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not ready yet. Check /health.")

    t_start = time.time()
    print(f"[api] /ocr/references START n_files={len(files)}", flush=True)

    run_dir = Path("/tmp/ocr_runs") / f"refs_{int(t_start)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    files_sorted = sorted(files, key=lambda f: (f.filename or ""))

    pages_out: list[dict] = []
    combined_text_parts: list[str] = []

    for idx, uf in enumerate(files_sorted, start=1):
        name = uf.filename or f"refpage_{idx}.png"
        img_bytes = await uf.read()

        if not img_bytes:
            pages_out.append({"source_file": name, "error": "Empty upload", "text": ""})
            continue

        try:
            pil = _decode_and_resize(img_bytes, max_side=2600)
        except Exception as e:
            pages_out.append({"source_file": name, "error": f"PIL decode failed: {e}", "text": ""})
            continue

        img_path = run_dir / name
        pil.save(img_path, format="PNG", optimize=True)

        print(f"[api] infer refs {name}", flush=True)

        try:
            t_page = time.time()
            async with _gpu_lock:
                with torch.inference_mode():
                    infer_res = model.infer(
                        tokenizer,
                        prompt=PROMPT_REFERENCES,
                        image_file=str(img_path),
                        output_path=str(run_dir),
                        base_size=1024,     # ✅ smaller for full pages
                        image_size=768,     # ✅ smaller for full pages
                        crop_mode=True,     # ✅ helps focus
                        save_results=False,
                        eval_mode=True,
                        test_compress=True, # ✅ faster/less memory
                    )
            print(f"[api] refs {name} infer_time={time.time()-t_page:.1f}s", flush=True)
        except Exception as e:
            pages_out.append({"source_file": name, "error": f"{type(e).__name__}: {e}", "text": ""})
            continue

        text = (_coerce_text(infer_res) or "").strip()
        pages_out.append({"source_file": name, "text": text})
        combined_text_parts.append(text)

    combined = "\n\n".join(combined_text_parts)
    combined = _clean_ref_text(combined)

    # ✅ If model says NONE anywhere, treat as not references page(s)
    # (when sending page-by-page you’ll typically get exactly "NONE")
    if combined.strip().upper() == "NONE":
        refs = ["NONE"]
        is_refs = False
    else:
        is_refs = _looks_like_bibliography(combined)
        if not is_refs:
            refs = ["NONE"]
        else:
            refs = _split_references(combined)
            # keep it safe
            refs = [r for r in refs if r and r.strip()]

    print(f"[api] /ocr/references DONE total_time={time.time()-t_start:.1f}s is_refs={is_refs} refs={len(refs) if refs != ['NONE'] else 0}", flush=True)

    return {
        "success": True,
        "files": [f.filename for f in files_sorted],
        "is_references": is_refs,
        "references_found": 0 if refs == ["NONE"] else len(refs),
        "references": refs,
        "combined_text": combined[:4000],  # small debug tail (optional)
        "pages": pages_out,
    }


def _clean_ref_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _remove_deepseek_tags(text: str) -> str:
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.S)
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.S)
    text = re.sub(r"<\|.*?\|>", "", text, flags=re.S)
    return text


def _strip_before_references(text: str) -> str:
    """
    Keep only content from the 'References' heading onward if present.
    """
    m = re.search(r"(?is)\breferences\b", text)
    if m:
        return text[m.end():].strip()
    return text.strip()


def _remove_obvious_non_reference_lines(text: str) -> str:
    bad_patterns = [
        r"(?im)^if you need to cite .*?$",
        r"(?im)^do not make any changes .*?$",
        r"(?im)^for more information about the publication.*?$",
        r"(?im)^##\s*7\.\s*conclusions.*?$",
        r"(?im)^author contributions:.*?$",
        r"(?im)^funding:.*?$",
        r"(?im)^institutional review board statement:.*?$",
        r"(?im)^informed consent statement:.*?$",
        r"(?im)^data availability statement:.*?$",
        r"(?im)^conflicts of interest:.*?$",
        r"(?im)^--- PAGE \d+ ---$",
    ]

    for pat in bad_patterns:
        text = re.sub(pat, "", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_references_from_deepseek_raw(text: str) -> list[dict]:
    """
    Parse numbered references from DeepSeek raw output.
    Supports:
      1. ...
      2. ...
      [1] ...
      1) ...
      (1) ...
    """
    text = _clean_ref_text(text)
    text = _remove_deepseek_tags(text)
    text = _strip_before_references(text)
    text = _remove_obvious_non_reference_lines(text)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    refs = []
    current_nr = None
    current_parts = []

    start_pat = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)\.|(\d+)\)|\((\d+)\))\s+(.*)$")

    for line in lines:
        m = start_pat.match(line)
        if m:
            if current_nr is not None and current_parts:
                ref_text = " ".join(current_parts).strip()
                ref_text = re.sub(r"\s+", " ", ref_text)
                refs.append({"nr": current_nr, "ref": ref_text})

            nr = next(g for g in m.groups()[:4] if g is not None)
            current_nr = int(nr)
            current_parts = [m.group(5).strip()]
        else:
            if current_nr is not None:
                # skip obvious garbage lines that sometimes appear in OCR
                if re.match(r"^\s*$", line):
                    continue
                current_parts.append(line)

    if current_nr is not None and current_parts:
        ref_text = " ".join(current_parts).strip()
        ref_text = re.sub(r"\s+", " ", ref_text)
        refs.append({"nr": current_nr, "ref": ref_text})

    # drop entries that are clearly not references
    cleaned = []
    for r in refs:
        ref = r["ref"].strip()
        if len(ref) < 15:
            continue
        if ref.lower().startswith("for example, root-feeding"):
            continue
        cleaned.append({"nr": r["nr"], "ref": ref})

    return cleaned



@app.post("/ocr/references2")
async def ocr_references2(
    file: UploadFile = File(...),
    ref_start_page_nr: int = Form(...)
):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    t_start = time.time()

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF upload")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        pdf_path = f.name

    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    doc.close()

    if ref_start_page_nr < 1 or ref_start_page_nr > n_pages:
        raise HTTPException(
            status_code=400,
            detail=f"ref_start_page_nr must be between 1 and {n_pages}"
        )

    run_dir = Path("/tmp/ocr_runs") / f"refs_{int(t_start)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pages_out = []
    raw_text_parts = []

    for page_nr in range(ref_start_page_nr, n_pages + 1):
        name = f"page_{page_nr:03d}.png"

        try:
            img_bytes = _render_pdf_page_to_png_bytes(pdf_path, page_nr - 1)
            pil = _decode_and_resize(img_bytes)
        except Exception as e:
            pages_out.append({
                "page_nr": page_nr,
                "error": f"render failed: {e}",
                "raw_text": "",
                "raw_obj_preview": ""
            })
            continue

        img_path = run_dir / name
        pil.save(img_path, format="PNG", optimize=True)

        try:
            async with _gpu_lock:
                with torch.inference_mode():
                    infer_res = model.infer(
                        tokenizer,
                        prompt=PROMPT_REFERENCES2,
                        image_file=str(img_path),
                        output_path=str(run_dir),
                        base_size=1024,
                        image_size=768,
                        crop_mode=True,
                        save_results=False,
                        eval_mode=True,
                        test_compress=True,
                    )
        except Exception as e:
            pages_out.append({
                "page_nr": page_nr,
                "error": f"{type(e).__name__}: {e}",
                "raw_text": "",
                "raw_obj_preview": ""
            })
            continue

        print("RAW:", infer_res)

        raw_text = (_coerce_text(infer_res) or "").strip()

        try:
            raw_obj_preview = repr(infer_res)[:1500]
        except Exception:
            raw_obj_preview = ""

        print(f"[DEBUG] page {page_nr} raw_text preview: {raw_text[:500]}", flush=True)
        print(f"[DEBUG] page {page_nr} infer_res repr: {raw_obj_preview[:500]}", flush=True)

        pages_out.append({
            "page_nr": page_nr,
            "raw_text": raw_text,
            "raw_obj_preview": raw_obj_preview
        })

        if raw_text:
            raw_text_parts.append(f"--- PAGE {page_nr} ---\n{raw_text}")

    combined_raw_text = "\n\n".join(raw_text_parts).strip()
    references_struct = _parse_references_from_deepseek_raw(combined_raw_text)

    return {
        "success": True,
        "file": file.filename,
        "pages_total": n_pages,
        "processed_pages": list(range(ref_start_page_nr, n_pages + 1)),
        "references_found": len(references_struct),
        "references": references_struct,
        "combined_raw_text": combined_raw_text,
        "pages": pages_out,
    }