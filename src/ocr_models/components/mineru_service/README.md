# MinerU Service

## Overview

This folder contains the MinerU-based table extraction service used in the pipeline.

The service receives the path of an uploaded PDF and an output directory from the backend. It then runs MinerU on the PDF, extracts table crops as PNG images, detects the start page of the references section, and writes table metadata to a JSON index file.

The service is exposed as a small FastAPI API and is started inside Docker.

---

# Purpose in the Pipeline

MinerU is used as the table detection and table image extraction component.

It is responsible for finding tables in scientific PDFs and saving them as cropped PNG files. These cropped table images are later sent to the PaddleOCR service for table structure recognition and cell extraction.

Pipeline position:

```text
PDF Upload
    ↓
Backend
    ↓
MinerU Service
    ↓
Cropped table PNGs
    ↓
PaddleOCR Service
```

---

# Folder Structure

```text
mineru_service/
├── app/
│   ├── main.py
│   └── table_extraction_benchmark/
│       └── runners/
│           └── mineru_tables_png_runner.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Main Files

## `app/main.py`

This file contains the FastAPI API for the MinerU service.

It provides:

| Endpoint         | Description                            |
| ---------------- | -------------------------------------- |
| `GET /health`    | Checks whether the service is running. |
| `POST /run-crop` | Runs MinerU table extraction on a PDF. |

The backend calls this endpoint after a PDF is uploaded.

---

## `mineru_tables_png_runner.py`

This file contains the MinerU execution logic.

It:

* starts the MinerU command-line pipeline,
* stores MinerU output in a job-specific folder,
* finds the generated `content_list.json`,
* detects table items inside the MinerU output,
* copies table images into the pipeline output folder,
* detects the start page of the references section,
* writes a `tables_index.json` file.

---

# API

## Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

---

## Run Table Crop Extraction

```text
POST /run-crop
```

Request body:

```json
{
  "pdf_path": "/app/data/uploads/example.pdf",
  "out_dir": "/app/data/processing/job_1"
}
```

Response:

```json
{
  "status": "done",
  "refs_start_page": 12,
  "tables_found": 5,
  "crops_saved": 5,
  "tables": []
}
```

---

# Workflow

## 1. Receive PDF Path from Backend

The backend sends the stored PDF path and job output directory to the MinerU service.

Example:

```text
pdf_path = /app/data/uploads/example.pdf
out_dir = /app/data/processing/job_1
```

---

## 2. Run MinerU

The runner executes MinerU with CUDA support.

The command is built internally similar to:

```bash
mineru -p <pdf_path> -o <mineru_out> -b Tabulus --device cuda -m ocr -t true -f false
```

The service uses GPU mode through:

```text
MINERU_DEVICE_MODE=cuda
```

---

## 3. Store MinerU Output

MinerU output is stored inside:

```text
/app/data/processing/job_<job_id>/mineru_out/
```

The service also writes:

```text
mineru_stdout.log
mineru_stderr.log
notes.md
```

These files are useful for debugging failed or incomplete runs.

---

## 4. Find `content_list.json`

After MinerU finishes, the service searches for the generated:

```text
*_content_list.json
```

This file contains structured information about detected content elements such as text blocks, headings, figures, and tables.

---

## 5. Detect References Start Page

The service scans the MinerU content list for headings such as:

* References
* Bibliography
* Literaturverzeichnis
* Quellen
* Referenzen

The last matching heading is used as the detected bibliography start page.

This value is stored as:

```text
refs_start_page
```

It is later used by the backend and reference matching step.

---

## 6. Extract Table PNGs

The service reads all items with:

```text
type == "table"
```

For each table item, it resolves the table image path and copies the PNG file into:

```text
/app/data/processing/job_<job_id>/images/tables/
```

The saved files are named like:

```text
page_003_table_001.png
page_004_table_002.png
```

---

## 7. Write Table Index

The service writes a metadata file:

```text
images/tables/tables_index.json
```

Example structure:

```json
{
  "tables_found": 5,
  "crops_saved": 5,
  "refs_start_page": 12,
  "tables": [
    {
      "table_id": 1,
      "page_nr": 3,
      "in_references": false,
      "png": "/app/data/processing/job_1/images/tables/page_003_table_001.png",
      "png_name": "page_003_table_001.png",
      "mineru_src": "...",
      "mineru_img_path": "...",
      "bbox": [],
      "table_caption": null,
      "table_footnote": null
    }
  ]
}
```

---

# Output Files

For each job, the MinerU service can create:

```text
job_<id>/
├── mineru_out/
├── mineru_stdout.log
├── mineru_stderr.log
├── notes.md
└── images/
    └── tables/
        ├── page_003_table_001.png
        ├── page_004_table_002.png
        └── tables_index.json
```

---

# Used Libraries

| Library      | Purpose                                         |
| ------------ | ----------------------------------------------- |
| `fastapi`    | Provides the MinerU service API.                |
| `uvicorn`    | Runs the FastAPI app.                           |
| `pydantic`   | Defines request models.                         |
| `requests`   | Included for service communication if needed.   |
| `mineru`     | Performs PDF parsing, OCR, and table detection. |
| `scipy`      | Required dependency in the Docker environment.  |
| `pathlib`    | Handles file paths.                             |
| `json`       | Reads and writes metadata files.                |
| `subprocess` | Runs the MinerU command-line pipeline.          |
| `shutil`     | Copies extracted table images.                  |
| `re`         | Detects reference section headings.             |

---

# Dockerfile

The service uses a PaddlePaddle GPU image because MinerU needs GPU-compatible dependencies.

```dockerfile
FROM paddlepaddle/paddle:3.3.0-gpu-cuda11.8-cudnn8.9

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    gcc \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libxkbcommon-x11-0 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    requests \
    scipy==1.15.3

RUN python3 -m pip install --no-cache-dir "mineru[all]"

COPY app /app/app

RUN mkdir -p /app/data/uploads /app/data/processing /app/data/results

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

---

# Requirements

```text
fastapi
uvicorn[standard]
pydantic
requests
mineru[core]
scipy==1.15.3
```

Note: The Dockerfile installs `mineru[all]`, while the requirements file lists `mineru[core]`. For Docker execution, the Dockerfile installation is the relevant one.

---

# Docker Compose Integration

The service is started by Docker Compose as:

```yaml
mineru_service:
  build:
    context: ./mineru_service
    dockerfile: Dockerfile
  container_name: mineru_service
  ports:
    - "8001:8001"
  volumes:
    - backend_data:/app/data
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [gpu]
```

Inside Docker, the backend reaches the service through:

```text
http://mineru_service:8001
```

From the host machine, the service is available at:

```text
http://localhost:8001
```

---

# Health Check

```text
http://localhost:8001/health
```

Expected response:

```json
{
  "status": "ok"
}
```

---

# Notes

This service was developed to extract table crops from scientific PDFs for later OCR processing.

The main goal is not to perform complete table recognition inside MinerU, but to reliably detect table regions and export them as PNG images.

These images are then processed by the OCR component of the pipeline.
