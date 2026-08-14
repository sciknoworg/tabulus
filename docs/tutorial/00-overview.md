# Core Pipeline Overview

The pipeline begins with one input scientific PDF paper and ends with resolved CSV files whose table reference cells are linked to DOI values where possible.

Each step should be implemented as a standalone processing component with a small, explicit contract. Libraries such as MinerU, PaddleOCR-VL, GROBID, Kreuzberg, Crossref, or future alternatives should be adapters behind these component boundaries.

## First Clean Workflow

This is the first practical workflow to stabilize: profile one scientific PDF with a plug-and-play document analysis adapter, currently MinerU, then reconstruct each table with PaddleOCR-VL.

```text
Scientific PDF
      |
      v

+----------------------+
| Module 1             |
| PDF Profiling        |
|                      |
| - page/layout        |
|   analysis           |
| - detect tables      |
| - determine table    |
|   bbox               |
| - save table crops   |
| - captions/footnotes |
| - structured JSON    |
+----------------------+
| Default adapter:     |
| MinerU 3.4.5         |
+----------------------+
      |
      | table images
      | structured JSON
      v

+----------------------+
| PaddleOCR-VL 1.6     |
| PaddleOCR 3.7.0      |
|                      |
| - read each table    |
| - recognize cells    |
| - understand         |
|   structure          |
| - reconstruct table  |
| - Markdown/structured|
|   output             |
+----------------------+
      |
      v

saved outputs
```

The first module is intentionally named for the processing work, not the library. MinerU is the first adapter for this module, but another PDF layout or table extraction tool should be able to produce the same outputs later.

The red output between the two modules is the important contract: table images plus structured metadata. In the current code, Tabulus does not crop the PDF from the bounding box itself. MinerU already generates the table crop image; Tabulus reads MinerU's `content_list.json`, filters entries with `type == "table"`, resolves each table's `img_path`, and copies that image into the run output.

Minimal standalone call using the current MinerU runner:

```python
from pathlib import Path

from app.table_extraction_benchmark.runners.mineru_tables_png_runner import run

pdf_path = Path("/data/runs/P51/input/paper.pdf")
run_dir = Path("/data/runs/P51")

run(pdf_path, run_dir)
```

This adapter writes table crops and structured metadata that later modules consume.

## Comparison To Evaluate

The clean workflow should compare two table extraction paths scientifically:

```text
MinerU table_body

versus

MinerU crop -> PaddleOCR-VL reconstruction
```

Modern MinerU may be good enough for some tables. The second model should not be assumed necessary until table quality, structure preservation, and runtime are measured.

## Ordered Steps

1. PDF profiling
2. Table OCR and structure extraction
3. Table normalization
4. Reference-table classification
5. Bibliography extraction
6. Reference matching
7. DOI resolution
8. Resolved CSV export
9. Run report and QA bundle

## Component Rule

Every step should be able to run in two modes:

- **Standalone mode:** process files from disk and write its own outputs.
- **Pipeline mode:** receive the previous step's contract output and return the next contract output.

## Current Code Mapping

| Pipeline step | Current implementation area | Notes |
| --- | --- | --- |
| PDF profiling | `src/Tabulus/mineru_service` | Uses MinerU to do page/layout analysis, table detection, crop image export, captions/footnotes, and structured JSON. |
| Table OCR | `src/Tabulus/paddleocr_service` | Uses PaddleOCR-VL and parses HTML or Markdown tables. |
| Reference-table classification | `src/Tabulus/backend/app/main.py` | Regex-based header and citation detection. |
| Bibliography extraction | `src/Tabulus/backend/app/reference_matching` | GROBID primary, Kreuzberg fallback. |
| Reference matching | `src/Tabulus/backend/app/reference_matching/grobid_reference_matching.py` | Numeric, DOI, author-year, author-only, text matching. |
| CSV export | `src/Tabulus/backend/app/reference_matching/grobid_reference_matching.py` | Writes resolved reference tables. |

## Tutorial Template

Every step page follows this structure:

- Goal
- Input
- Output
- Module contract
- Default implementation
- Alternative adapters
- Standalone run target
- Verification
- Common failure modes
- Next step
