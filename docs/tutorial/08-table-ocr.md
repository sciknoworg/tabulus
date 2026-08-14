# Step 8: Table OCR And Structure Extraction

## Goal

Turn table crop images into structured rows and columns.

## Input

`tables/tables_index.json` and table crop PNGs.

## Output

`tables/ocr_tables.json`.

## Module Contract

See `data-contracts/ocr-tables-json.md`.

## Default Implementation

The current implementation sends table PNGs to PaddleOCR-VL and parses HTML or Markdown tables from the model output.

In the first clean workflow, the expected adapter stack is:

```text
table crop PNGs
      |
      v
PaddleOCR-VL 1.6 / PaddleOCR 3.7.0
      |
      v
Markdown or structured table output
      |
      v
tables/ocr_tables.json
```

## Alternative Adapters

- PaddleOCR-VL
- DeepSeek OCR
- Chandra OCR
- Kreuzberg OCR
- NuExtract3

## Verification

The step succeeds when each crop has an OCR result object, even if the OCR result is empty or contains an error.
