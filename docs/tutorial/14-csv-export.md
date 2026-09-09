# Planned Stage 7: Resolved CSV Export

## Goal

Join resolved scholarly identities back onto relevant table cells and write
final resolved CSV files for reference-like tables.

## Input

The planned input is prediction CSV or parsed table rows, Stage 5 reference
matches, and the Stage 6 paper-level resolution registry.

## Output

```text
resolved_reference_tables/
  page_003_table_001_resolved.csv
```

## Module Contract

See {doc}`../data-contracts/resolved-csv`.

## Default Implementation

This stage is retained in the legacy thesis workflow but is not yet
implemented in the rebuilt `src/tabulus` library.

The target exporter must keep two CSV concepts separate:

- prediction CSV: reconstructed table before reference resolution or DOI
  enrichment; used for table-quality evaluation
- resolved CSV: final user-facing table after bibliography matching and
  paper-level scholarly resolution

Stage 7 is planned as a deterministic join and export step. Every relevant
table cell/reference occurrence of the same bibliography index should receive
the same resolved paper-level identity. Exact export
columns, status semantics, and rejected-link handling remain part of the future
export contract.

Resolved CSV files are intended per relevant/reference-containing table.
Original reference values and provenance should remain traceable in downstream
export artifacts.

## Verification

The future step should be verified by checking that validated paper-level
identities are joined consistently to all relevant occurrences and exported
without changing reconstruction prediction CSVs.
