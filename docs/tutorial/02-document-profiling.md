# Step 2: Document Profiling

## Goal

Inspect the ingested PDF before expensive processing begins.

## Input

`metadata/ingestion.json` and `input/paper.pdf`.

## Output

`metadata/document_profile.json`.

## Module Contract

```json
{
  "run_id": "P51",
  "page_count": 14,
  "has_extractable_text": true,
  "text_sample": "...",
  "metadata": {
    "title": null,
    "author": null
  },
  "status": "profiled"
}
```

## Default Implementation

Use `PyMuPDF` or `pypdf` to read page count and lightweight document metadata. Use extracted text from the first few pages as a signal, not as final OCR.

## Alternative Adapters

- `PyMuPDF`
- `pypdf`
- `pdfplumber`

## Verification

The step succeeds when page count is known and the next step can decide which pages to render.
