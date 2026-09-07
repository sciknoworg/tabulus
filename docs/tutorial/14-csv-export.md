# Planned Stage 7: Resolved CSV Export

## Goal

Join validated scholarly identities back onto relevant table cells and write
final resolved CSV files for reference-like tables.

## Input

Prediction CSV or parsed table rows, Stage 5 reference matches, and the
Stage 6 paper-level `references/reference_resolution.json` registry.

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

Stage 7 is planned as a deterministic join and export step. Every relevant
table cell/reference occurrence of the same bibliography index should receive
the same validated Stage 6 identity. `validated_without_doi` is a legitimate
identity and must not be treated as a rejected reference merely because no DOI
is established. Exact export columns and rejected-link handling remain part
of the future export contract.

Resolved CSV files are intended per relevant/reference-containing table.
Original reference values and provenance should remain traceable in the
downstream export artifacts.

## Verification

The future step should be verified by checking that validated paper-level
identities are joined consistently to all relevant occurrences and exported
without changing reconstruction prediction CSVs.
