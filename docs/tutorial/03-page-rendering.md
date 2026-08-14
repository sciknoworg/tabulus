:orphan:

# PDF Profiling Internal: Page Rendering

## Goal

Convert PDF pages into page images for downstream layout detection and OCR components.

## Input

`input/paper.pdf` plus `metadata/pdf_profile.json`.

## Output

```text
pages/
  page_001.png
  page_002.png
metadata/page_rendering.json
```

## Module Contract

```json
{
  "dpi": 200,
  "pages_rendered": 14,
  "images": [
    {
      "page_nr": 1,
      "path": "pages/page_001.png",
      "width": 1700,
      "height": 2200
    }
  ],
  "status": "rendered"
}
```

## Default Implementation

Use `PyMuPDF` for deterministic page rendering.

## Verification

The step succeeds when every expected page image exists and has non-zero dimensions.
