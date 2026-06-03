# OCR Model Evaluation Components

This folder contains the OCR model services and runner scripts used during the evaluation phase of the project.

The final Tabulus pipeline is located in `src/Tabulus/` and contains a dedicated README describing the complete pipeline workflow.

---

## Folder Structure

```text
ocr_models/
├── components/
│   ├── deepseekOCR2/
│   ├── Kreuzberg/
│   ├── mineru_service/
│   └── paddleOCR_VL/
│
├── KISSKI/
│   └── Chandra/
│       └── bin/
│
└── runners/
    ├── services/
    ├── runner_chandra_png_to_csv.py
    ├── runner_chandra_raw_references.py
    ├── runner_kreuzberg_raw_references.py
    ├── runner_kreuzberg_table_png.py
    ├── runner_mineru_deepseek_tables_and_refs.py
    ├── runner_mineru_paddle_tables.py
    └── runner_paddle_references.py
```

---

## Python Environment

The runner scripts were developed and tested using a Python virtual environment.

Example setup:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Depending on the OCR component, additional model-specific dependencies may be required inside the corresponding Docker containers.

---

## Docker Components

Each OCR component can be started separately from its own folder.

---

### MinerU Service

Used to detect and crop tables from PDF files.

```bash
cd src/ocr_models/components/mineru_service
docker compose up --build
```

The MinerU service is used by several runners to create table PNG crops before another OCR model processes them.

---

### DeepSeek OCR 2

Used for table OCR and optional reference extraction.

```bash
cd src/ocr_models/components/deepseekOCR2
docker compose up --build
```

Expected API:

```text
http://127.0.0.1:8000
```

Used by:

```text
runner_mineru_deepseek_tables_and_refs.py
```

---

### PaddleOCR-VL

Used for table OCR and reference extraction.

```bash
cd src/ocr_models/components/paddleOCR_VL
docker compose up --build
```

Expected API:

```text
http://127.0.0.1:8002
```

Used by:

```text
runner_mineru_paddle_tables.py
runner_paddle_references.py
```

---

### Kreuzberg

Used for raw text extraction from PDFs and OCR extraction from table PNG images.

```bash
cd src/ocr_models/components/Kreuzberg
docker compose up --build
```

Expected API:

```text
http://127.0.0.1:8010
```

Used by:

```text
runner_kreuzberg_raw_references.py
runner_kreuzberg_table_png.py
```

---

## Runner Scripts

Run all runner scripts from:

```bash
cd src/ocr_models
```

This is important because the imports are based on this working directory.

---

### MinerU + DeepSeek Tables and References

This runner first uses MinerU to crop table images from the PDF and can optionally send the generated table crops to DeepSeek OCR 2.

```bash
python runners/runner_mineru_deepseek_tables_and_refs.py \
  --pdf ./components/deepseekOCR2/data/input/P1.pdf \
  --out ./components/deepseekOCR2/data/output/P1
```

With DeepSeek table OCR:

```bash
python runners/runner_mineru_deepseek_tables_and_refs.py \
  --pdf ./components/deepseekOCR2/data/input/P1.pdf \
  --out ./components/deepseekOCR2/data/output/P1 \
  --deepseek-tables
```

With table OCR and reference extraction:

```bash
python runners/runner_mineru_deepseek_tables_and_refs.py \
  --pdf ./components/deepseekOCR2/data/input/P1.pdf \
  --out ./components/deepseekOCR2/data/output/P1 \
  --deepseek-tables \
  --deepseek-refs
```

---

### MinerU + PaddleOCR-VL Tables

This runner first uses MinerU to crop tables and then sends the cropped table images to PaddleOCR-VL.

```bash
python runners/runner_mineru_paddle_tables.py \
  --pdf ./components/paddleOCR_VL/data/input/P1.pdf \
  --out ./components/paddleOCR_VL/data/output/P1
```

---

### PaddleOCR-VL References

This runner sends a PDF directly to the PaddleOCR-VL reference extraction endpoint.

```bash
python runners/runner_paddle_references.py \
  --pdf ./components/paddleOCR_VL/data/input/P1.pdf \
  --ref-start-page 13 \
  --out ./components/paddleOCR_VL/data/output/P1/Ref
```

---

### Kreuzberg Raw References

This runner sends a PDF to Kreuzberg and extracts the raw reference section starting from a given page.

```bash
python runners/runner_kreuzberg_raw_references.py \
  --pdf ./components/Kreuzberg/data/input/P1.pdf \
  --start-page 13 \
  --out ./components/Kreuzberg/data/output/P1/Ref
```

---

### Kreuzberg Table PNG OCR

This runner sends a single table PNG image to Kreuzberg and saves the extracted table as CSV, Markdown, and JSON.

```bash
python runners/runner_kreuzberg_table_png.py \
  --png ./components/Kreuzberg/data/input/page_001_table_001.png \
  --out ./components/Kreuzberg/data/output/P1/Kreuzberg_prediction
```

---

## Chandra on KISSKI

Chandra was executed remotely on the KISSKI cluster via SSH and Slurm.

The required Slurm scripts are located in:

```text
KISSKI/Chandra/bin/
├── pdf_to_txt_sbatch.sh
└── png_to_md_sbatch.sh
```

These scripts are not Docker-based.

---

### Chandra Table PNG to CSV

This runner uploads a PNG table image to KISSKI, submits a Slurm job, downloads the Markdown output, and converts it to CSV.

```bash
python runners/runner_chandra_png_to_csv.py \
  --png ./components/Chandra/data/input/page_001_table_001.png \
  --out ./components/Chandra/data/output/P1
```

---

### Chandra Raw References

This runner uploads a PDF to KISSKI and submits a Slurm job for raw reference text extraction.

```bash
python runners/runner_chandra_raw_references.py \
  --pdf ./components/Chandra/data/input/P1/P1.pdf \
  --start-page 13
```

The result is created remotely on the KISSKI system.

---

## Notes

* The OCR components are evaluation and benchmarking components.
* They are not part of the final Tabulus pipeline.
* The final pipeline is located in `src/Tabulus/`.
* `src/Tabulus/README.md` describes the complete production workflow.
* Large datasets and generated outputs should not be committed to the repository.

Recommended ignored folders:

```text
data/
work/
logs/
cache/
env/
__pycache__/
*.pyc
```
