# Step 4: Table OCR And Structure Extraction

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

PaddleOCR-VL is more than ordinary OCR. Its current architecture performs layout analysis followed by vision-language-model recognition. The layout stage detects elements such as tables, crops them, determines reading order, and the VLM converts the elements into structured recognition results.

In Tabulus, MinerU has already isolated the table image during PDF profiling. PaddleOCR-VL therefore receives a cleaner input than it would receive from a full page.

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

## Evaluation Question

The table OCR module should be evaluated against MinerU's own structured table output:

```text
MinerU table_body

versus

MinerU crop -> PaddleOCR-VL reconstruction
```

Modern MinerU may be sufficient for some table classes. The second model should remain a measured choice, not an assumption.

## Verification

The step succeeds when each crop has an OCR result object, even if the OCR result is empty or contains an error.
