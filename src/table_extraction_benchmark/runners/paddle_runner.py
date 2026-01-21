from __future__ import annotations

import os
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import cv2
import numpy as np


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class OCRLine:
    text: str
    confidence: float
    poly: List[List[int]]      # 4-pt poly in PAGE coords
    bbox_xyxy: List[int]       # [x1,y1,x2,y2] in PAGE coords
    reading_order: int = 0     # order inside its block


@dataclass
class LayoutBlock:
    block_id: int
    label: str                 # table/text/figure/chart/...
    score: float
    bbox_xyxy: List[int]
    lines: List[OCRLine]


# -----------------------------
# IO helpers
# -----------------------------
def _safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_write_csv(path: Path, rows: List[List[Any]], header: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(rows)


# -----------------------------
# Geometry helpers
# -----------------------------
def _poly_to_xyxy(poly: List[List[int]]) -> List[int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _center_xyxy(b: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _clip_xyxy(b: List[int], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _point_in_xyxy(pt: Tuple[float, float], box: List[int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)


# -----------------------------
# Render PDF
# -----------------------------
def _render_page_to_bgr(pdf_path: Path, page_index: int, dpi: int = 200) -> np.ndarray:
    doc = fitz.open(str(pdf_path))
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise ValueError(f"page_index {page_index} out of range (0..{doc.page_count-1})")
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cv2.imdecode failed")
        return img
    finally:
        doc.close()


# -----------------------------
# Reading order (fast heuristic)
# -----------------------------
def _reading_order_heuristic(lines: List[OCRLine], y_tol: int = 12) -> List[int]:
    """
    Compute reading order positions for these lines.
    Group by y-center into rows (tolerance), then x sort.
    Returns list where result[i] is reading position for lines[i].
    """
    items = []
    for i, ln in enumerate(lines):
        cx, cy = _center_xyxy(ln.bbox_xyxy)
        items.append((i, cx, cy))
    items.sort(key=lambda t: (t[2], t[1]))

    rows: List[List[Tuple[int, float, float]]] = []
    for it in items:
        if not rows:
            rows.append([it])
            continue
        _, _, cy = it
        last_row_cy = float(np.mean([x[2] for x in rows[-1]]))
        if abs(cy - last_row_cy) <= y_tol:
            rows[-1].append(it)
        else:
            rows.append([it])

    ordered_idx: List[int] = []
    for r in rows:
        r.sort(key=lambda t: t[1])
        ordered_idx.extend([t[0] for t in r])

    pos = {idx: p for p, idx in enumerate(ordered_idx)}
    return [pos[i] for i in range(len(lines))]


# -----------------------------
# Plot 1: OCR text detection (green boxes + blue text)
# -----------------------------
def _draw_ocr_text(img: np.ndarray, blocks: List[LayoutBlock], max_label_len: int = 45) -> np.ndarray:
    out = img.copy()
    for b in blocks:
        for ln in b.lines:
            pts = np.array(ln.poly, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)

            label = (ln.text or "").strip()
            if not label:
                continue
            if len(label) > max_label_len:
                label = label[: max_label_len - 3] + "..."

            x1, y1, _, _ = ln.bbox_xyxy
            tx = max(0, x1)
            ty = max(15, y1 - 6)
            cv2.putText(out, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
    return out


# -----------------------------
# Plot 2: OCR reading order (green boxes + red order numbers)
# -----------------------------
def _draw_ocr_order(img: np.ndarray, blocks: List[LayoutBlock]) -> np.ndarray:
    out = img.copy()
    for b in blocks:
        for ln in b.lines:
            pts = np.array(ln.poly, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)

            cx, cy = _center_xyxy(ln.bbox_xyxy)
            cv2.putText(out, str(int(ln.reading_order)), (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return out


# -----------------------------
# Plot 3: Layout areas (magenta boxes + label)
# -----------------------------
def _draw_layout_areas(img: np.ndarray, blocks: List[LayoutBlock]) -> np.ndarray:
    out = img.copy()
    for b in blocks:
        x1, y1, x2, y2 = b.bbox_xyxy
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        cv2.polylines(out, [pts], True, (255, 0, 255), 2)
        label = f"{b.block_id}:{b.label} {b.score:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    return out


# -----------------------------
# Table CSV export (simple row grouping)
# -----------------------------
def _group_rows_for_csv(lines: List[OCRLine], y_tol: int = 12) -> List[List[OCRLine]]:
    if not lines:
        return []

    def y_center(ln: OCRLine) -> float:
        x1, y1, x2, y2 = ln.bbox_xyxy
        return (y1 + y2) / 2.0

    sorted_lines = sorted(lines, key=lambda ln: (ln.bbox_xyxy[1], ln.bbox_xyxy[0]))

    rows: List[List[OCRLine]] = []
    for ln in sorted_lines:
        cy = y_center(ln)
        if not rows:
            rows.append([ln])
            continue
        last = rows[-1]
        last_cy = float(np.mean([y_center(x) for x in last]))
        if abs(cy - last_cy) <= y_tol:
            last.append(ln)
        else:
            rows.append([ln])

    for r in rows:
        r.sort(key=lambda ln: ln.bbox_xyxy[0])
    return rows


def _export_tables_csv(out_dir: Path, page_number: int, blocks: List[LayoutBlock]) -> None:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    table_blocks = [b for b in blocks if b.label.lower() == "table"]
    for i, tb in enumerate(table_blocks, start=1):
        rows = _group_rows_for_csv(tb.lines, y_tol=12)
        csv_rows = [[cell.text for cell in row] for row in rows]
        path = tables_dir / f"page_{page_number:03d}_table_{i:02d}.csv"
        _safe_write_csv(path, rows=csv_rows, header=None)


# -----------------------------
# NEW: Full-page OCR (better text quality)
# -----------------------------
def _run_full_page_ocr(page_img_bgr: np.ndarray) -> Tuple[List[OCRLine], Optional[np.ndarray]]:
    """
    Run PaddleOCR ON THE FULL PAGE (not cropped).
    Returns:
      - list of OCRLine with poly/bbox in page coords
      - optional processed image from doc_preprocessor (for better-aligned visualization)
    """
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="en", use_angle_cls=True)

    try:
        result = ocr.predict(page_img_bgr)  # numpy input
    except Exception:
        # fallback: caller can write to disk and pass path; here we just retry with the same
        result = ocr.predict(page_img_bgr)

    page = result[0] if result else {}

    texts = page.get("rec_texts", []) or []
    scores = page.get("rec_scores", []) or []
    polys = page.get("rec_polys") or page.get("dt_polys") or []

    lines: List[OCRLine] = []
    for t, s, p in zip(texts, scores, polys):
        poly = np.array(p, dtype=np.int32).reshape(-1, 2).tolist()
        bbox = _poly_to_xyxy(poly)
        lines.append(OCRLine(
            text=str(t),
            confidence=float(s) if s is not None else 0.0,
            poly=poly,
            bbox_xyxy=bbox,
            reading_order=0,
        ))

    processed = page.get("doc_preprocessor_res", {}).get("output_img")
    processed_img = processed if isinstance(processed, np.ndarray) else None

    return lines, processed_img


# -----------------------------
# NEW: Assign OCR lines to layout blocks (no crop OCR)
# -----------------------------
def _assign_lines_to_blocks(all_lines: List[OCRLine], blocks: List[LayoutBlock]) -> None:
    """
    Assign each OCR line to the best matching block.
    For robustness (especially tables), use center-point containment.
    If not inside any block -> unassigned.
    """
    unassigned = None

    for b in blocks:
        b.lines = []

    for ln in all_lines:
        cx, cy = _center_xyxy(ln.bbox_xyxy)

        chosen: Optional[LayoutBlock] = None
        for b in blocks:
            if _point_in_xyxy((cx, cy), b.bbox_xyxy):
                chosen = b
                break

        if chosen is None:
            if unassigned is None:
                # create one unassigned block spanning whole page area
                # bbox will be filled by caller if needed; keep as empty placeholder here
                unassigned = LayoutBlock(
                    block_id=max(b.block_id for b in blocks) + 1 if blocks else 0,
                    label="unassigned",
                    score=1.0,
                    bbox_xyxy=[0, 0, 10, 10],
                    lines=[],
                )
                blocks.append(unassigned)
            chosen = unassigned

        chosen.lines.append(ln)


# -----------------------------
# Core processing:
#   layout detection -> full-page OCR -> assign lines to blocks -> order within blocks
# -----------------------------
def _process_one_page(
    pdf_path: Path,
    out_dir: Path,
    page_number: int,
    dpi: int,
    min_layout_score: float,
    crop_padding: int,      # kept for API compatibility; not used in full-page OCR mode
    ocr_labels: List[str],  # kept to filter which blocks are kept (optional)
) -> Dict[str, Any]:
    # Ensure env flags are set early for paddlex/paddleocr internals
    os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_use_onednn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["FLAGS_enable_new_executor"] = "0"

    from paddleocr import LayoutDetection  # import after env

    pages_dir = out_dir / "pages"
    plots_dir = out_dir / "plots"
    raw_dir = out_dir / "raw"
    pages_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1) Render page to PNG (for record/debug)
    page_idx = page_number - 1
    page_img = _render_page_to_bgr(pdf_path, page_idx, dpi=dpi)
    h, w = page_img.shape[:2]
    page_png = pages_dir / f"page_{page_number:03d}.png"
    cv2.imwrite(str(page_png), page_img)

    # 2) Layout detection (table/text/figure/chart etc.)
    layout_engine = LayoutDetection()
    layout_out = layout_engine.predict(str(page_png))
    layout_boxes = (layout_out[0].get("boxes", []) if layout_out else []) or []

    blocks: List[LayoutBlock] = []
    block_id = 0
    for b in layout_boxes:
        label = str(b.get("label", "unknown"))
        score = float(b.get("score", 0.0))
        if score < min_layout_score:
            continue
        coord = b.get("coordinate", None)
        if not coord or len(coord) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in coord]
        bbox = _clip_xyxy([x1, y1, x2, y2], w, h)
        blocks.append(LayoutBlock(block_id=block_id, label=label, score=score, bbox_xyxy=bbox, lines=[]))
        block_id += 1

    if not blocks:
        blocks = [LayoutBlock(block_id=0, label="document", score=1.0, bbox_xyxy=[0, 0, w, h], lines=[])]

    # Optional: keep only blocks with certain labels (but DO NOT delete them if you want full plot 3)
    # For your use-case: keep them all, but OCR_LABELS can be used later if desired.
    # (So we keep this as a no-op.)

    # 3) Full-page OCR (better detection)
    all_lines, processed_img = _run_full_page_ocr(page_img)

    # 4) Assign OCR lines to blocks (no crop OCR)
    # Also fix unassigned bbox if created
    _assign_lines_to_blocks(all_lines, blocks)
    for b in blocks:
        if b.label == "unassigned":
            b.bbox_xyxy = [0, 0, w, h]

    # 5) Reading order within each block (fast heuristic)
    for b in blocks:
        if not b.lines:
            continue
        ro = _reading_order_heuristic(b.lines, y_tol=12)
        for i, ln in enumerate(b.lines):
            ln.reading_order = int(ro[i])
        b.lines.sort(key=lambda x: x.reading_order)

    # Sort blocks by page position
    blocks.sort(key=lambda b: (b.bbox_xyxy[1], b.bbox_xyxy[0]))

    # Choose visualization base image:
    # processed_img aligns with OCR coords and usually looks nicer for OCR plot
    vis_base = processed_img if isinstance(processed_img, np.ndarray) else page_img

    # 3 plots per page
    plot1 = _draw_ocr_text(vis_base, blocks)          # AFTER text detection (OCR)
    plot2 = _draw_ocr_order(vis_base, blocks)         # AFTER order
    plot3 = _draw_layout_areas(page_img, blocks)      # FINAL: layout areas (on original page render)

    cv2.imwrite(str(plots_dir / f"page_{page_number:03d}_1_ocr_text.png"), plot1)
    cv2.imwrite(str(plots_dir / f"page_{page_number:03d}_2_order.png"), plot2)
    cv2.imwrite(str(plots_dir / f"page_{page_number:03d}_3_layout.png"), plot3)

    # JSON output
    out_json: Dict[str, Any] = {
        "pdf": str(pdf_path),
        "page_number": page_number,
        "dpi": dpi,
        "ocr_labels_used": ocr_labels,
        "blocks": [
            {
                "block_id": b.block_id,
                "label": b.label,
                "score": b.score,
                "bbox_xyxy": b.bbox_xyxy,
                "lines": [asdict(ln) for ln in b.lines],
            }
            for b in blocks
        ],
    }
    _safe_write_json(raw_dir / f"page_{page_number:03d}_blocks.json", out_json)

    # table CSVs
    _export_tables_csv(out_dir, page_number, blocks)

    print(
        f"[paddle_layout] page {page_number} blocks={len(blocks)} "
        f"ocr_lines(full_page)={len(all_lines)}"
    )
    return out_json


# -----------------------------
# Runner entry point for run_one.py
# -----------------------------
def run(pdf_path: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Default: process only page 5
    - env PADDLE_LAYOUT_PAGE=<n>        -> one specific page
    - env PADDLE_LAYOUT_ALL_PAGES=1     -> all pages

    Other knobs:
    - env PADDLE_LAYOUT_OCR_LABELS="table,chart,text,figure"
    - env PADDLE_LAYOUT_DPI="220"
    - env PADDLE_LAYOUT_CROP_PADDING="10"   (kept; not used in full-page OCR mode)
    - env PADDLE_LAYOUT_MIN_LAYOUT_SCORE="0.50"
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(os.getenv("PADDLE_LAYOUT_DPI", "200"))
    min_layout_score = float(os.getenv("PADDLE_LAYOUT_MIN_LAYOUT_SCORE", "0.50"))
    crop_padding = int(os.getenv("PADDLE_LAYOUT_CROP_PADDING", "10"))

    labels_str = os.getenv("PADDLE_LAYOUT_OCR_LABELS", "table,chart,text,figure")
    ocr_labels = [s.strip().lower() for s in labels_str.split(",") if s.strip()]

    default_page = 5
    env_page = os.getenv("PADDLE_LAYOUT_PAGE")
    env_all = os.getenv("PADDLE_LAYOUT_ALL_PAGES", "0")

    doc = fitz.open(str(pdf_path))
    try:
        total_pages = doc.page_count
    finally:
        doc.close()

    if env_all.strip() == "1":
        pages = list(range(1, total_pages + 1))
    elif env_page and env_page.strip().isdigit():
        pages = [int(env_page.strip())]
    else:
        pages = [default_page]

    summary: Dict[str, Any] = {"pdf": str(pdf_path), "pages_processed": pages, "results": []}

    for p in pages:
        if p < 1 or p > total_pages:
            print(f"[paddle_layout] skipping invalid page {p} (pdf has {total_pages})")
            continue
        summary["results"].append(_process_one_page(
            pdf_path=pdf_path,
            out_dir=out_dir,
            page_number=p,
            dpi=dpi,
            min_layout_score=min_layout_score,
            crop_padding=crop_padding,
            ocr_labels=ocr_labels,
        ))

    _safe_write_json(out_dir / "raw" / "summary.json", summary)
    print(f"[paddle_layout] processed pages {pages} -> {out_dir}")
    return summary
