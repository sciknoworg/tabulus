# Step 9: Table Normalization

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

This is not yet cleanly separated in the code. Some normalization happens inside the PaddleOCR service while parsing HTML or Markdown.

## Verification

The step succeeds when rows are rectangular enough for later column detection and CSV export.
