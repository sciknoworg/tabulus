# Step 5: Table Normalization

## Goal

Clean OCR table rows while preserving the raw OCR result for debugging.

## Input

`tables/ocr_tables.json`.

## Output

`tables/normalized_tables.json`.

## Module Contract

```json
{
  "tables": [
    {
      "table_id": 1,
      "source_file": "page_003_table_001.png",
      "rows": [["Header", "Value"], ["A", "B"]],
      "normalization_warnings": []
    }
  ],
  "status": "tables_normalized"
}
```

## Default Implementation

Full scientific table normalization is not yet implemented as a standalone stage.

The current `tabulus.table_ocr.parsing` layer restores the legacy Paddle-compatible row parser. It reads adapter-native Markdown/HTML text, prefers HTML `<table>...</table>` elements when present, falls back to GitHub-style pipe-table Markdown only when no HTML table is found, and returns a rectangular row representation with:

- `rows`
- `n_rows`
- `n_cols`
- `source`

This parser is a reconstruction/parsing checkpoint, not final normalization. It does not semantically fill down cells, interpret section rows, rewrite formulas, merge continued tables, resolve references, or decide which reconstruction candidate is scientifically correct.

## Verification

The step succeeds when rows are rectangular enough for later column detection and CSV export.
