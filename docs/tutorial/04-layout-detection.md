:orphan:

# PDF Profiling Internal: Document Layout Detection

## Goal

Detect high-level document elements such as headings, paragraphs, figures, captions, and tables.

In the first clean workflow, this work belongs to the combined **Page Layout And Table Crop Extraction** module. That module performs:

- page/layout analysis
- table detection
- table bounding-box detection
- table crop export
- caption and footnote capture
- structured JSON export

## Input

Rendered page images and/or the original PDF.

## Output

`layout/layout_items.json`.

## Module Contract

```json
{
  "items": [
    {
      "type": "table",
      "page_nr": 3,
      "bbox": [100, 200, 900, 600],
      "text": null,
      "confidence": null,
      "source": "mineru"
    }
  ],
  "status": "layout_detected"
}
```

## Default Implementation

The current pipeline gets layout information from MinerU's `content_list.json`.

The current standalone adapter call is:

```python
from pathlib import Path

from app.table_extraction_benchmark.runners.mineru_tables_png_runner import run

run(
    pdf_path=Path("/data/runs/P51/input/paper.pdf"),
    out_dir=Path("/data/runs/P51"),
)
```

## Alternative Adapters

- MinerU
- Docling
- Nougat-style layout outputs
- Custom detector over page images

## Verification

The step succeeds when layout items include page numbers and normalized item types.
