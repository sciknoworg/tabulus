# Step 4: Table OCR And Structure Extraction

## Goal

Turn table crop images into structured rows and columns.

## Input

`tables_index.json` and table crop images from the table-crop handoff directory.

The input is intentionally image-based. Table-reconstruction adapters consume canonical MinerU-generated table crops from the normalized Tabulus handoff; they do not independently process the original PDF to detect and crop tables for this comparison.

## Output

`tables/ocr_tables.json`.

## Module Contract

See `data-contracts/ocr-tables-json.md`.

## Default Implementation

This step is not yet implemented in the new Tabulus library. The new modular pipeline should treat it as a later adapter stage after PDF profiling and table-crop export are stable.

The component is model-independent: a Table OCR and Structure Extraction adapter consumes the normalized Tabulus table-crop handoff and returns a structured table result. PaddleOCR-VL is the first/default adapter being implemented for this contract, but another table-reconstruction adapter can be substituted later if it accepts the same handoff and preserves the same MinerU provenance.

PaddleOCR-VL is more than ordinary OCR. Its current architecture performs layout analysis followed by vision-language-model recognition. The layout stage detects elements such as tables, crops them, determines reading order, and the VLM converts the elements into structured recognition results.

In the intended Tabulus workflow, MinerU has already isolated the table image during PDF profiling. PaddleOCR-VL therefore receives a cleaner input than it would receive from a full page and is evaluated as a table-reconstruction adapter, not as a competing full-document table detector.

The expected handoff from PDF profiling is a table-crop collection with image paths plus provenance: page number, bounding box, caption, footnote, original MinerU `img_path`, and MinerU `table_body` when available.

Conceptually, the adapter contract is:

```text
MinerU table crop
       |
       v
Table reconstruction adapter
       |
       v
structured table
```

It is not:

```text
original PDF
       |
       v
each OCR adapter independently detects and crops tables
```

The adapter stage should focus on extracting or reconstructing cell text, rows, columns, table structure, and adapter-native structured output while preserving the table ID and MinerU provenance supplied by the normalized handoff.

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

The diagram above shows the first adapter implementation, not a restriction of the component contract.

## Alternative Adapters

- PaddleOCR-VL -- first/default implementation target
- DeepSeek OCR
- Chandra
- Kreuzberg
- NuExtract3

These are alternative table-reconstruction adapters, not sequential pipeline stages.

## Evaluation Question

The initial validation should preserve the first clean comparison against MinerU's own structured table output:

```text
MinerU table_body

versus

MinerU crop -> PaddleOCR-VL reconstruction
```

Modern MinerU may be sufficient for some table classes. The second model should remain a measured choice, not an assumption.

For each table, keep both outputs during evaluation:

- MinerU `table_body`
- PaddleOCR-VL reconstruction from the MinerU-generated crop image

The extended adapter benchmark is the more general version of the same question:

```text
MinerU table_body
versus
MinerU crop -> PaddleOCR-VL
versus
MinerU crop -> DeepSeek OCR
versus
MinerU crop -> Chandra
versus
MinerU crop -> Kreuzberg
versus
MinerU crop -> NuExtract3
```

PaddleOCR-VL remains the first implementation target even though the architecture supports multiple adapters. The pipeline can later decide whether to use the lighter MinerU output directly for some table classes or route crops through a table-reconstruction adapter.

## Verification

The step succeeds when each crop has an OCR result object, even if the OCR result is empty or contains an error.
