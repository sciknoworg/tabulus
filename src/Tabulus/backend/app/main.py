import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from app.reference_matching.grobid_reference_matching import (
    match_reference_tables_with_grobid,
    write_resolved_reference_table_csvs,
)

GROBID_URL = os.getenv("GROBID_URL", "http://grobid:8070")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")

KREUZBERG_API_URL = os.getenv(
    "KREUZBERG_API_URL",
    "http://kreuzberg:8010/extract",
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/data/temp.db")
MINERU_API_URL = os.getenv("MINERU_API_URL", "http://mineru_service:8001")
PADDLEOCR_API_URL = os.getenv("PADDLEOCR_API_URL", "http://paddleocr_service:8000")

BASE_DIR = Path("/app/data")
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSING_DIR = BASE_DIR / "processing"
RESULTS_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PipelineJob(Base):
    __tablename__ = "pipeline_jobs"

    id = Column(Integer, primary_key=True, index=True)
    original_name = Column(String, nullable=False)
    stored_pdf_path = Column(String, nullable=False)
    processing_dir = Column(String, nullable=False)
    status = Column(String, nullable=False, default="uploaded")
    refs_start_page = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TAG_PATTERN = re.compile(
    r"\b("
    r"refs?\.?|references?|"
    r"auth(?:\b|ors?\b)|"
    r"years?|"
    r"sources?|"
    r"papers?|"
    r"citations?|"
    r"literatures?|"
    r"works?|"
    r"comparative\s+works?|"
    r"research(?:es)?|"
    r"datasets?|data\s*sets?|"
    r"data\s*set(?:\s*(?:name|naem))?|"
    r"dataset(?:\s*(?:name|naem))?|"
    r"publications?|pubications?|"
    r"contributions?|"
    r"sample\s+articles?|"
    r"stud(?:y|ies)(?!\s*area)"
    r")\b",
    re.IGNORECASE,
)

CITATION_PATTERN = re.compile(
    r"("
    r"\[\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?(?:\s*,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?)*\s*\]"
    r"|"
    r"\(\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?(?:\s*,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?)*\s*\)"
    r"|"
    r"\b[A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+)?\s+et\s+al\.?.*?(?:19|20)\d{2}[a-z]?\b"
    r"|"
    r"\b[A-Z][A-Za-z'`\-]+,\s*(?:19|20)\d{2}[a-z]?\b"
    r"|"
    r"\bdoi\s*:\s*10\.\S+\b"
    r"|"
    r"\b10\.\S+\b"
    r"|"
    r"\b[A-Z][A-Za-z'`\-]+\s+and\s+[A-Z][A-Za-z'`\-]+.*?(?:19|20)\d{2}[a-z]?\b"
    r")",
    re.IGNORECASE,
)


def normalize_text(text):

    return re.sub(r"\s+", " ", str(text)).strip()


def get_first_non_empty_cells_by_column(rows):
    if not rows:
        return []

    max_cols = max(len(row) for row in rows)
    first_non_empty_cells = []

    for col_idx in range(max_cols):
        first_value = ""

        for row in rows:
            cell = row[col_idx] if col_idx < len(row) else ""
            cell = normalize_text(cell)

            if cell:
                first_value = cell
                break

        first_non_empty_cells.append(first_value)

    return first_non_empty_cells


def classify_reference_like_table(rows):
    if not rows:
        return {
            "is_reference_table": False,
            "has_tag_match": False,
            "has_citation_match": False,
            "matched_header_cells": [],
            "matched_citation_cells": [],
            "reason": "No rows available.",
        }

    normalized_rows = [[normalize_text(cell) for cell in row] for row in rows]
    first_non_empty_cells = get_first_non_empty_cells_by_column(normalized_rows)

    matched_header_cells = [
        cell for cell in first_non_empty_cells if cell and TAG_PATTERN.search(cell)
    ]

    matched_citation_cells = []

    for row in normalized_rows:
        for cell in row:
            if cell and CITATION_PATTERN.search(cell):
                matched_citation_cells.append(cell)

                if len(matched_citation_cells) >= 5:
                    break

        if len(matched_citation_cells) >= 5:
            break

    has_tag_match = len(matched_header_cells) > 0
    has_citation_match = len(matched_citation_cells) > 0

    is_reference_table = False
    reason = "No reference-like evidence found."

    if has_tag_match and has_citation_match:
        is_reference_table = True
        reason = "Header-like reference tags and citation-like cell content found."
    elif not has_tag_match and has_citation_match:
        is_reference_table = True
        reason = "No header tag found, but citation-like cell content found."
    elif has_tag_match and not has_citation_match:
        reason = "Header-like tags found, but no citation-like content found."

    return {
        "is_reference_table": is_reference_table,
        "has_tag_match": has_tag_match,
        "has_citation_match": has_citation_match,
        "matched_header_cells": matched_header_cells[:5],
        "matched_citation_cells": matched_citation_cells[:5],
        "reason": reason,
    }


def annotate_ocr_tables_with_reference_detection(paddle_data):
    tables = paddle_data.get("tables", []) or []

    enriched_tables = []
    reference_tables_found = 0

    for table in tables:
        rows = table.get("rows", []) or []
        detection = classify_reference_like_table(rows)

        enriched = {
            **table,
            **detection,
        }

        if detection["is_reference_table"]:
            reference_tables_found += 1

        enriched_tables.append(enriched)

    return {
        **paddle_data,
        "tables": enriched_tables,
        "reference_tables_found": reference_tables_found,
    }


def send_table_crops_to_paddle(job_out_dir: Path):
    index_path = job_out_dir / "images" / "tables" / "tables_index.json"

    if not index_path.exists():
        return {
            "success": True,
            "tables_found": 0,
            "tables": [],
        }

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    tables = index_data.get("tables", [])

    files = []
    opened_files = []

    try:
        for table in tables:
            png_name = table.get("png_name")

            if not png_name:
                continue

            img_path = job_out_dir / "images" / "tables" / png_name

            if not img_path.exists():
                continue

            f = img_path.open("rb")
            opened_files.append(f)
            files.append(("files", (png_name, f, "image/png")))

        if not files:
            return {
                "success": True,
                "tables_found": 0,
                "tables": [],
            }

        response = requests.post(
            f"{PADDLEOCR_API_URL}/ocr/images",
            files=files,
            timeout=60 * 30,
        )

        response.raise_for_status()
        return response.json()

    finally:
        for f in opened_files:
            f.close()


def load_tables_index(job_out_dir: Path):
    index_path = job_out_dir / "images" / "tables" / "tables_index.json"

    if not index_path.exists():
        return {
            "refs_start_page": None,
            "tables_found": 0,
            "crops_saved": 0,
            "tables": [],
        }

    data = json.loads(index_path.read_text(encoding="utf-8"))

    return {
        "refs_start_page": data.get("refs_start_page"),
        "tables_found": data.get("tables_found", 0),
        "crops_saved": data.get("crops_saved", 0),
        "tables": data.get("tables", []),
    }


def get_ocr_result_path(job_out_dir: Path) -> Path:
    return job_out_dir / "ocr_tables.json"


@app.get("/")
def root():
    return {"message": "Backend is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "database": DATABASE_URL,
        "upload_dir": str(UPLOAD_DIR),
        "processing_dir": str(PROCESSING_DIR),
        "mineru_api_url": MINERU_API_URL,
        "paddleocr_api_url": PADDLEOCR_API_URL,
        "grobid_url": GROBID_URL,
        "kreuzberg_api_url": KREUZBERG_API_URL,
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    safe_original_name = Path(file.filename).name
    stored_name = f"{uuid.uuid4()}_{safe_original_name}"
    pdf_target_path = UPLOAD_DIR / stored_name

    with pdf_target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db = SessionLocal()

    try:
        job = PipelineJob(
            original_name=safe_original_name,
            stored_pdf_path=str(pdf_target_path),
            processing_dir="",
            status="uploaded",
        )

        db.add(job)
        db.commit()
        db.refresh(job)

        job_out_dir = PROCESSING_DIR / f"job_{job.id}"
        job_out_dir.mkdir(parents=True, exist_ok=True)

        job.processing_dir = str(job_out_dir)
        job.status = "mineru_running"
        db.commit()

        mineru_response = requests.post(
            f"{MINERU_API_URL}/run-crop",
            json={
                "pdf_path": str(pdf_target_path),
                "out_dir": str(job_out_dir),
            },
            timeout=60 * 30,
        )

        if mineru_response.status_code != 200:
            job.status = "mineru_error"
            db.commit()

            raise HTTPException(
                status_code=500,
                detail=f"MinerU service failed: {mineru_response.text}",
            )

        mineru_data = mineru_response.json()

        refs_start_page = mineru_data.get("refs_start_page")

        if isinstance(refs_start_page, int):
            job.refs_start_page = refs_start_page

        job.status = "mineru_done"
        db.commit()

        return {
            "job_id": job.id,
            "status": job.status,
            "original_name": job.original_name,
            "stored_pdf_path": job.stored_pdf_path,
            "processing_dir": job.processing_dir,
            "refs_start_page": mineru_data.get("refs_start_page"),
            "tables_found": mineru_data.get("tables_found", 0),
            "crops_saved": mineru_data.get("crops_saved", 0),
            "tables": mineru_data.get("tables", []),
            "ocr_tables_found": 0,
            "ocr_tables": [],
            "reference_tables_found": 0,
        }

    except requests.RequestException as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Could not reach service: {str(e)}")

    except HTTPException:
        raise

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Upload or MinerU processing failed: {str(e)}",
        )

    finally:
        db.close()


@app.post("/jobs/{job_id}/run-paddle")
def run_paddle(job_id: int):
    db = SessionLocal()

    try:
        job = db.query(PipelineJob).filter(PipelineJob.id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")

        job_out_dir = Path(job.processing_dir)

        if not job_out_dir.exists():
            raise HTTPException(status_code=404, detail="Processing directory not found.")

        index_data = load_tables_index(job_out_dir)

        if index_data["tables_found"] == 0:
            raise HTTPException(
                status_code=400,
                detail="No cropped table images found for this job.",
            )

        job.status = "ocr_running"
        db.commit()

        paddle_data = send_table_crops_to_paddle(job_out_dir)
        paddle_data = annotate_ocr_tables_with_reference_detection(paddle_data)

        ocr_result_path = get_ocr_result_path(job_out_dir)
        ocr_result_path.write_text(
            json.dumps(paddle_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        job.status = "done"
        db.commit()

        return {
            "job_id": job.id,
            "status": job.status,
            "original_name": job.original_name,
            "stored_pdf_path": job.stored_pdf_path,
            "processing_dir": job.processing_dir,
            "refs_start_page": index_data.get("refs_start_page"),
            "tables_found": index_data.get("tables_found", 0),
            "crops_saved": index_data.get("crops_saved", 0),
            "tables": index_data.get("tables", []),
            "ocr_tables_found": paddle_data.get("tables_found", 0),
            "ocr_tables": paddle_data.get("tables", []),
            "reference_tables_found": paddle_data.get("reference_tables_found", 0),
        }

    except requests.RequestException as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Could not reach PaddleOCR service: {str(e)}",
        )

    except HTTPException:
        raise

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"PaddleOCR processing failed: {str(e)}",
        )

    finally:
        db.close()


@app.get("/jobs/{job_id}")
def get_job(job_id: int):
    db = SessionLocal()

    try:
        job = db.query(PipelineJob).filter(PipelineJob.id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")

        job_out_dir = Path(job.processing_dir)
        index_data = load_tables_index(job_out_dir)

        ocr_result_path = get_ocr_result_path(job_out_dir)
        ocr_data = {
            "tables_found": 0,
            "tables": [],
            "reference_tables_found": 0,
        }

        if ocr_result_path.exists():
            ocr_data = json.loads(ocr_result_path.read_text(encoding="utf-8"))

        return {
            "job_id": job.id,
            "original_name": job.original_name,
            "stored_pdf_path": job.stored_pdf_path,
            "processing_dir": job.processing_dir,
            "status": job.status,
            "refs_start_page": job.refs_start_page,
            "created_at": job.created_at.isoformat(),
            "tables_found": index_data.get("tables_found", 0),
            "crops_saved": index_data.get("crops_saved", 0),
            "tables": index_data.get("tables", []),
            "ocr_tables_found": ocr_data.get("tables_found", 0),
            "ocr_tables": ocr_data.get("tables", []),
            "reference_tables_found": ocr_data.get("reference_tables_found", 0),
        }

    finally:
        db.close()


@app.get("/jobs/{job_id}/images/{image_name}")
def get_table_image(job_id: int, image_name: str):
    image_path = PROCESSING_DIR / f"job_{job_id}" / "images" / "tables" / image_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(image_path)


@app.post("/jobs/{job_id}/match-references")
def match_references(
    job_id: int,
    use_crossref: bool = False,
    use_kreuzberg_fallback: bool = True,
):
    db = SessionLocal()

    try:
        job = db.query(PipelineJob).filter(PipelineJob.id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")

        pdf_path = Path(job.stored_pdf_path)

        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="Stored PDF not found.")

        job_out_dir = Path(job.processing_dir)
        ocr_result_path = get_ocr_result_path(job_out_dir)

        if not ocr_result_path.exists():
            raise HTTPException(
                status_code=400,
                detail="OCR result not found. Run PaddleOCR first.",
            )

        ocr_data = json.loads(ocr_result_path.read_text(encoding="utf-8"))
        ocr_tables = ocr_data.get("tables", [])

        job.status = "reference_matching_running"
        db.commit()

        match_data = match_reference_tables_with_grobid(
            pdf_path=pdf_path,
            ocr_tables=ocr_tables,
            grobid_url=GROBID_URL,
            crossref_mailto=CROSSREF_MAILTO,
            use_crossref=use_crossref,
            kreuzberg_api_url=KREUZBERG_API_URL if use_kreuzberg_fallback else None,
            refs_start_page=job.refs_start_page,
        )

        resolved_csvs = write_resolved_reference_table_csvs(
            job_out_dir=job_out_dir,
            ocr_tables=ocr_tables,
            match_data=match_data,
        )

        match_data["resolved_csvs"] = resolved_csvs

        result_path = job_out_dir / "reference_matches.json"
        result_path.write_text(
            json.dumps(match_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        job.status = "reference_matching_done"
        db.commit()

        return {
            "job_id": job.id,
            "status": job.status,
            **match_data,
        }

    except HTTPException:
        raise

    except Exception as e:
        db.rollback()

        job = db.query(PipelineJob).filter(PipelineJob.id == job_id).first()

        if job:
            job.status = "reference_matching_error"
            db.commit()

        raise HTTPException(
            status_code=500,
            detail=f"Reference matching failed: {str(e)}",
        )

    finally:
        db.close()


@app.get("/jobs/{job_id}/resolved-csv/{csv_name}")
def get_resolved_csv(job_id: int, csv_name: str):
    job_dir = PROCESSING_DIR / f"job_{job_id}"
    matches = list(job_dir.rglob(csv_name))

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"Resolved CSV not found: {csv_name}",
        )

    return FileResponse(
        matches[0],
        media_type="text/csv",
        filename=csv_name,
    )