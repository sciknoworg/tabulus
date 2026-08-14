# PDF Readers

PDF readers inspect the PDF without doing OCR-heavy work.

## Responsibility

- Read page count.
- Read embedded metadata when available.
- Sample embedded text.
- Flag likely scanned/image-only documents.

## Candidate Tools

- `PyMuPDF`
- `pypdf`
- `pdfplumber`

The reader output becomes `metadata/document_profile.json`.
