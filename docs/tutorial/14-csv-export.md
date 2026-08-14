# Step 14: Resolved CSV Export

## Goal

Write final CSV files for reference-like tables.

## Input

OCR table rows and reference matches.

## Output

```text
resolved_reference_tables/
  page_003_table_001_resolved.csv
```

## Module Contract

See `data-contracts/resolved-csv.md`.

## Default Implementation

The current exporter renames the detected reference column to `DOI` and replaces matched values only when DOI values are available.

## Verification

The step succeeds when every matched reference-like table has a downloadable CSV output.
