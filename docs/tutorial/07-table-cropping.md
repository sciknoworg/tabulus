:orphan:

# PDF Profiling Internal: Table Cropping

## Goal

Create one image file per detected table.

In the first clean workflow, table cropping is part of the combined **Page Layout And Table Crop Extraction** module.

## Input

Detected tables and page images, or detector-provided table images.

## Output

```text
work/table_crops/
  tables_index.json
  images/
    page_003_table_001.png
```

## Module Contract

See `data-contracts/tables-index-json.md`.

## Default Implementation

The current implementation copies table images referenced by MinerU `content_list.json` into the normalized table-crop handoff directory.

The expected module output is:

```text
work/table_crops/
  tables_index.json
  images/
    page_003_table_001.png
```

Run it with:

```bash
tabulus export-table-crops --mineru-root work/mineru/puurunen_2005 --out work/table_crops
```

## Verification

The step succeeds when every indexed crop path exists and can be opened as an image.
