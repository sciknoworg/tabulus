# Backend Service

## Overview

This folder contains the FastAPI backend used to orchestrate the scientific PDF processing pipeline.

The backend acts as the central coordination component between:

* the frontend upload interface,
* the MinerU table extraction service,
* the PaddleOCR table recognition service,
* the GROBID bibliography extraction service,
* the Kreuzberg OCR fallback service,
* and optional Crossref DOI enrichment.

The backend manages the complete workflow from PDF upload to resolved reference table CSV generation.

---

# System Architecture

The backend is responsible for communication between all pipeline services.

```text
Frontend UI
    ↓
FastAPI Backend
    ↓
MinerU Table Extraction
    ↓
PaddleOCR Table Recognition
    ↓
Reference-like Table Detection
    ↓
GROBID Bibliography Extraction
    ↓
Kreuzberg Fallback (optional)
    ↓
Reference Matching
    ↓
Resolved CSV Export
```

---

# Main Responsibilities

The backend performs the following tasks:

1. Receive uploaded PDF files from the frontend.
2. Store uploaded PDFs in a persistent Docker volume.
3. Create and track pipeline jobs in a SQLite database.
4. Create processing folders for each uploaded document.
5. Call the MinerU service to detect and crop tables from PDFs.
6. Send cropped table images to the PaddleOCR service.
7. Detect reference-like tables using regex-based heuristics.
8. Extract bibliography entries with GROBID.
9. Use Kreuzberg as fallback when GROBID extraction fails.
10. Match extracted table references against bibliography entries.
11. Resolve DOI values.
12. Export resolved reference tables as CSV files.
13. Store intermediate and final outputs as JSON files.

---

# Folder Structure

```text
backend/
├── app/
│   ├── main.py
│   └── reference_matching/
│       ├── grobid_reference_matching.py
│       └── kreuzberg_reference_fallback.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Main Components

## `app/main.py`

This file contains:

* the FastAPI application,
* REST API endpoints,
* SQLite job management,
* communication with external services,
* OCR orchestration,
* and reference table classification.

### Main API Endpoints

| Endpoint                                     | Description                                          |
| -------------------------------------------- | ---------------------------------------------------- |
| `GET /`                                      | Verifies that the backend is running.                |
| `GET /health`                                | Returns backend configuration and service URLs.      |
| `POST /upload-pdf`                           | Uploads a PDF and starts MinerU processing.          |
| `POST /jobs/{job_id}/run-paddle`             | Runs OCR on cropped table images.                    |
| `GET /jobs/{job_id}`                         | Returns job state and pipeline results.              |
| `GET /jobs/{job_id}/images/{image_name}`     | Returns a cropped table image.                       |
| `POST /jobs/{job_id}/match-references`       | Runs bibliography extraction and reference matching. |
| `GET /jobs/{job_id}/resolved-csv/{csv_name}` | Downloads a resolved reference table CSV file.       |

---

## `grobid_reference_matching.py`

This module contains the main bibliography extraction and reference matching logic.

The module is responsible for:

* TEI XML parsing,
* DOI extraction,
* numeric reference matching,
* author-year matching,
* author-only matching,
* Crossref enrichment,
* reference column detection,
* bibliography extraction,
* and CSV export.

### Supported Matching Strategies

The matching pipeline supports multiple reference styles:

#### Numeric references

Examples:

```text
[1]
[3,4]
[5-8]
```

#### Author-year references

Examples:

```text
Smith et al. 2020
Smith and Jones 2019
Smith, 2020
```

#### DOI references

Examples:

```text
10.1016/j.tplants.2018.02.001
https://doi.org/10.xxxx
```

#### Author-only fallback matching

Used when incomplete references appear inside the table.

---

## `kreuzberg_reference_fallback.py`

This module implements the fallback bibliography extraction logic.

The fallback pipeline is activated when:

* GROBID fails,
* GROBID returns no bibliography,
* or the extracted bibliography is unusable.

The module:

* extracts raw OCR text from the PDF,
* isolates the references section,
* removes OCR and page noise,
* and applies publisher/style-specific regex patterns.

### Supported bibliography styles

| Pattern   | Description                            |
| --------- | -------------------------------------- |
| Numbered  | `[1]`, `1.` style references           |
| APA       | Author-year bibliography style         |
| Springer  | Springer-like bibliography structure   |
| Wiley     | Wiley journal reference patterns       |
| Frontiers | Frontiers publisher bibliography style |
| BES       | British Ecological Society style       |

Currently, the fallback pipeline accepts numbered bibliography extraction as the primary fallback strategy.

---

# Database

The backend uses SQLite for lightweight pipeline job management.

## PipelineJob Table

| Field             | Description                          |
| ----------------- | ------------------------------------ |
| `id`              | Unique pipeline job identifier       |
| `original_name`   | Original uploaded PDF filename       |
| `stored_pdf_path` | Stored PDF path inside Docker volume |
| `processing_dir`  | Processing output directory          |
| `status`          | Current pipeline state               |
| `refs_start_page` | Detected bibliography start page     |
| `created_at`      | Job creation timestamp               |

---

# Used Libraries

| Library            | Purpose                             |
| ------------------ | ----------------------------------- |
| `fastapi`          | Backend REST API                    |
| `uvicorn`          | ASGI application server             |
| `sqlalchemy`       | SQLite database management          |
| `requests`         | HTTP communication between services |
| `python-multipart` | PDF file uploads                    |
| `pydantic`         | Request and response validation     |
| `lxml`             | GROBID TEI XML parsing              |
| `re`               | Regex-based pattern matching        |
| `json`             | Intermediate result storage         |
| `csv`              | CSV export generation               |
| `pathlib`          | File system path handling           |
| `uuid`             | Unique uploaded file names          |
| `datetime`         | Timestamp generation                |

---

# Processing Workflow

## Step 1 — PDF Upload

The frontend uploads a PDF using:

```text
POST /upload-pdf
```

The backend:

* validates the file,
* stores it in the uploads directory,
* creates a database job entry,
* creates a processing folder,
* and starts MinerU table extraction.

Stored location:

```text
/app/data/uploads/
```

---

## Step 2 — Table Detection with MinerU

The backend calls:

```text
http://mineru_service:8001/run-crop
```

MinerU:

* detects tables inside PDF pages,
* crops table regions,
* stores cropped PNG images,
* and creates `tables_index.json` metadata.

Example output:

```text
/app/data/processing/job_1/images/tables/
```

---

## Step 3 — OCR with PaddleOCR

The endpoint:

```text
POST /jobs/{job_id}/run-paddle
```

loads cropped table images and sends them to:

```text
http://paddleocr_service:8000/ocr/images
```

The OCR output is saved as:

```text
ocr_tables.json
```

The OCR result contains:

* extracted table rows,
* row/column metadata,
* source image information,
* and reference-table classification results.

---

## Step 4 — Reference-like Table Detection

After OCR, the backend classifies whether a table likely contains references.

The detection uses two signals:

### Header-based detection

Matches terms such as:

* references
* authors
* citations
* papers
* datasets
* publications
* studies

### Citation-based detection

Matches patterns such as:

```text
[1]
(2)
Smith et al. 2020
Smith, 2020
10.xxxx
```

The classifier enriches OCR tables with:

* matched headers,
* matched citation cells,
* detection reason,
* and reference-table status.

---

## Step 5 — Bibliography Extraction

The endpoint:

```text
POST /jobs/{job_id}/match-references
```

starts bibliography extraction.

### Primary extraction method: GROBID

The backend calls:

```text
http://grobid:8070/api/processReferences
```

GROBID returns TEI XML.

The backend parses the XML using `lxml` and extracts:

* bibliography entries,
* DOI values,
* and reference indices.

---

## Step 6 — Kreuzberg Fallback

If GROBID fails or returns no bibliography entries, the backend optionally uses Kreuzberg OCR.

The Kreuzberg fallback pipeline:

1. extracts raw OCR text,
2. isolates the references section,
3. removes publisher and OCR noise,
4. applies bibliography regex patterns,
5. validates extracted references.

The fallback currently focuses on numbered bibliography patterns.

---

## Step 7 — Reference Matching

The backend attempts to match OCR table references against extracted bibliography entries.

Matching strategies include:

* numeric reference matching,
* DOI matching,
* author-year matching,
* author-only matching,
* text containment matching.

Optional Crossref enrichment can be enabled to resolve missing DOI values.

---

## Step 8 — CSV Export

Resolved reference tables are exported as CSV files.

The detected reference column is replaced with DOI values when matches are found.

If no match is found, the cell is cleared.

Output location:

```text
/app/data/processing/job_1/resolved_reference_tables/
```

---

# Generated Outputs

Each pipeline job creates a dedicated processing folder:

```text
/app/data/processing/job_<job_id>/
```

Typical outputs include:

```text
ocr_tables.json
reference_matches.json
resolved_reference_tables/
images/tables/
tables_index.json
```

---

# Environment Variables

| Variable            | Default                         | Description             |
| ------------------- | ------------------------------- | ----------------------- |
| `DATABASE_URL`      | `sqlite:////app/data/temp.db`   | SQLite database path    |
| `MINERU_API_URL`    | `http://mineru_service:8001`    | MinerU service URL      |
| `PADDLEOCR_API_URL` | `http://paddleocr_service:8000` | PaddleOCR service URL   |
| `GROBID_URL`        | `http://grobid:8070`            | GROBID service URL      |
| `KREUZBERG_API_URL` | `http://kreuzberg:8010/extract` | Kreuzberg service URL   |
| `CROSSREF_MAILTO`   | empty                           | Optional Crossref email |

---

# Docker Configuration

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /app/data/uploads /app/data/processing /app/data/results

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Requirements

```text
fastapi
uvicorn[standard]
sqlalchemy
requests
python-multipart
pydantic
lxml
```

---

# Running the Backend

From the main pipeline directory:

```bash
docker compose up --build
```

The backend becomes available on:

```text
http://localhost:8000
```

Health endpoint:

```text
http://localhost:8000/health
```

---

# Notes

This backend was developed as part of a scientific pipeline for extracting tables and resolving references from scientific PDFs.

The system is designed for Docker-based local execution and research artifact demonstration.

The implementation focuses on:

* modular service communication,
* reproducible processing,
* OCR benchmarking,
* reference-table detection,
* bibliography extraction,
* and DOI resolution.
