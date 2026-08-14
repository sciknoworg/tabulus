# PDF Ingestors

PDF ingestors register input papers as pipeline runs.

## Responsibility

- Validate that the input exists and is a PDF.
- Create the run directory.
- Copy or link the source PDF.
- Write `metadata/ingestion.json`.

## Default Tooling

Use Python standard library tools:

- `pathlib`
- `shutil`
- `hashlib`

Use `pypdf` or `PyMuPDF` only if the ingestor also performs a minimal open/read validation.

## Adapter Ideas

- Single file ingestor
- Folder ingestor
- Manifest ingestor
- Future web-upload ingestor
