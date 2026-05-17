import json
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


from app.table_extraction_benchmark.runners.mineru_tables_png_runner import run as mineru_tables_png_run

app = FastAPI()


class RunCropRequest(BaseModel):
    pdf_path: str
    out_dir: str

PROCESSING_DIR = Path("/app/data/processing")
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run-crop")
def run_crop(req: RunCropRequest):
    pdf_path = Path(req.pdf_path)
    out_dir = Path(req.out_dir)

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        mineru_tables_png_run(pdf_path, out_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MinerU crop failed: {str(e)}")

    index_path = out_dir / "images" / "tables" / "tables_index.json"

    if not index_path.exists():
        return {
            "status": "done_no_index",
            "refs_start_page": None,
            "tables_found": 0,
            "crops_saved": 0,
            "tables": [],
        }

    index_data = json.loads(index_path.read_text(encoding="utf-8"))

    return {
        "status": "done",
        "refs_start_page": index_data.get("refs_start_page"),
        "tables_found": index_data.get("tables_found", 0),
        "crops_saved": index_data.get("crops_saved", 0),
        "tables": index_data.get("tables", []),
    }

