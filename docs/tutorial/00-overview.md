# Core Pipeline Overview

The pipeline begins with one input scientific PDF paper and ends with resolved CSV files whose table reference cells are linked to DOI values where possible.

Each step should be implemented as a standalone processing component with a small, explicit contract. Libraries such as MinerU, PaddleOCR-VL, GROBID, Kreuzberg, Crossref, or future alternatives should be adapters behind these component boundaries.

## Ordered Steps

1. PDF ingestion
2. Document profiling
3. Page rendering
4. Document layout detection
5. Reference section detection
6. Table detection
7. Table cropping
8. Table OCR and structure extraction
9. Table normalization
10. Reference-table classification
11. Bibliography extraction
12. Reference matching
13. DOI resolution
14. Resolved CSV export
15. Run report and QA bundle

## Component Rule

Every step should be able to run in two modes:

- **Standalone mode:** process files from disk and write its own outputs.
- **Pipeline mode:** receive the previous step's contract output and return the next contract output.

## Current Code Mapping

| Pipeline step | Current implementation area | Notes |
| --- | --- | --- |
| PDF ingestion | `src/Tabulus/backend/app/main.py` | Currently tied to FastAPI upload. Needs a local file-folder adapter. |
| Table detection and cropping | `src/Tabulus/mineru_service` | Uses MinerU and writes table PNGs plus `tables_index.json`. |
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
