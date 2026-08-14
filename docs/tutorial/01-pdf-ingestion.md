# Step 1: PDF Ingestion

## Goal

Register one scientific PDF paper as a reproducible pipeline run.

PDF ingestion is not OCR. It is the step that validates the input file, assigns a run id, creates the run directory, copies or links the source PDF, and writes the first metadata record.

## Tool Choice

The current application uses FastAPI's `UploadFile` plus Python file handling in `src/Tabulus/backend/app/main.py`.

For the modular non-Docker pipeline, the ingestion component should use standard Python file tools:

- `pathlib` for paths
- `shutil` for copying the PDF into a run directory
- `hashlib` for a stable file checksum
- `pypdf` or `PyMuPDF` only for lightweight validation and page-count checks

This means the first tool is not MinerU or PaddleOCR. The first tool is a small local Python ingestion module that reads an existing PDF from your folder of papers and creates a clean run folder.

## Input

One PDF file from a local papers folder.

Example:

```text
papers/
  P51.pdf
```

## Output

A run directory with the source PDF and ingestion metadata.

```text
runs/
  P51/
    input/
      paper.pdf
    metadata/
      ingestion.json
```

## Module Contract

The ingestion component should write `metadata/ingestion.json`.

```json
{
  "run_id": "P51",
  "original_path": "C:/path/to/papers/P51.pdf",
  "stored_pdf_path": "runs/P51/input/paper.pdf",
  "original_filename": "P51.pdf",
  "sha256": "...",
  "file_size_bytes": 1234567,
  "ingested_at": "2026-08-14T12:00:00Z",
  "status": "ingested"
}
```

## Default Implementation

The current backend implementation does part of this during `POST /upload-pdf`, but it is coupled to the web API and SQLite job table.

The recommended standalone component should be a small command:

```powershell
python -m tabulus_pipeline.ingest_pdf `
  --pdf C:\path\to\papers\P51.pdf `
  --runs-root C:\path\to\runs
```

Expected result:

```text
runs/P51/input/paper.pdf
runs/P51/metadata/ingestion.json
```

## Alternative Adapters

- Single local PDF file
- Folder of PDFs
- CSV manifest of paper ids and PDF paths
- Future upload endpoint

All adapters should produce the same `ingestion.json` contract.

## Standalone Run Target

The first useful development target is:

```text
Given a folder of papers, create one run directory per PDF and write ingestion metadata for each paper.
```

## Verification

The step succeeds when:

- The PDF exists in `input/paper.pdf`.
- `metadata/ingestion.json` exists.
- The checksum is present.
- The status is `ingested`.
- A later step can read `stored_pdf_path`.

## Common Failure Modes

| Failure | Likely cause | Fix |
| --- | --- | --- |
| File not found | Wrong path or manifest entry | Validate paths before processing. |
| Not a PDF | Extension or header mismatch | Reject during ingestion. |
| Duplicate run id | Two files have same stem | Add collision handling or manifest ids. |
| Cannot copy file | Permission or disk issue | Fail before later pipeline work starts. |

## Next Step

After ingestion, run document profiling to determine page count, embedded text availability, and whether the paper appears scanned or text-based.
