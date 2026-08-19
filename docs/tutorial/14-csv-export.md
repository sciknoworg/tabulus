# Step 10: Resolved CSV Export

## Goal

Write final resolved CSV files for reference-like tables.

## Input

Prediction CSV or normalized table rows, plus reference matches and DOI values.

## Output

```text
resolved_reference_tables/
  page_003_table_001_resolved.csv
```

## Module Contract

See `data-contracts/resolved-csv.md`.

## Default Implementation

This stage is retained in the legacy thesis workflow but is not yet implemented in the rebuilt `src/tabulus` library.

The target exporter must keep two CSV concepts separate:

- prediction CSV: reconstructed table before reference resolution or DOI enrichment; used for RMS/DePlot table-quality evaluation
- resolved CSV: final user-facing table after bibliography matching and DOI resolution

Resolved CSV files are produced per relevant/reference-containing table, not one per PDF. The detected reference column should be renamed to `DOI`; when DOI values are found, they replace the original reference value, and when no DOI is found, the original value remains traceable.

## Verification

The step succeeds when every matched reference-like table has a downloadable CSV output.
