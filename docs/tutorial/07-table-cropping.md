# Step 7: Table Cropping

## Goal

Create one image file per detected table.

## Input

Detected tables and page images, or detector-provided table images.

## Output

```text
tables/crops/
  page_003_table_001.png
tables/tables_index.json
```

## Module Contract

See `data-contracts/tables-index-json.md`.

## Default Implementation

The current MinerU runner copies table images referenced by MinerU `content_list.json` into `images/tables/`.

## Verification

The step succeeds when every indexed crop path exists and can be opened as an image.
