# Step 4: Table OCR And Structure Extraction

## Goal

Turn table crop images into structured rows and columns.

## Input

`tables_index.json` and table crop images from the table-crop handoff directory.

## Output

`tables/ocr_tables.json`.

## Module Contract

See `data-contracts/ocr-tables-json.md`.

## Default Implementation

This step is not yet implemented in the new Tabulus library.

The legacy service code contains PaddleOCR-VL integration work, but the new modular pipeline should treat this as a later adapter stage after PDF profiling and table-crop export are stable.

PaddleOCR-VL is more than ordinary OCR. Its current architecture performs layout analysis followed by vision-language-model recognition. The layout stage detects elements such as tables, crops them, determines reading order, and the VLM converts the elements into structured recognition results.

In the intended Tabulus workflow, MinerU has already isolated the table image during PDF profiling. PaddleOCR-VL therefore receives a cleaner input than it would receive from a full page.

The expected handoff from PDF profiling is a table-crop collection with image paths plus provenance: page number, bounding box, caption, footnote, original MinerU `img_path`, and MinerU `table_body` when available.

In the first clean workflow, the expected adapter stack is:

```text
table crop images
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

For each table, keep both outputs during evaluation:

- MinerU `table_body`
- PaddleOCR-VL reconstruction from the MinerU-generated crop image

The pipeline can later decide whether to use the lighter MinerU output directly for some table classes.

## Verification

The step succeeds when each crop has an OCR result object, even if the OCR result is empty or contains an error.
