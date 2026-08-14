# Step 4: Document Layout Detection

## Goal

Detect high-level document elements such as headings, paragraphs, figures, captions, and tables.

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

## Alternative Adapters

- MinerU
- Docling
- Nougat-style layout outputs
- Custom detector over page images

## Verification

The step succeeds when layout items include page numbers and normalized item types.
