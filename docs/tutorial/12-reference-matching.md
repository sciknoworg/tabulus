# Planned Stage: Reference Matching

## Goal

Match table reference cells to bibliography entries.

## Input

Classified reference-like tables from the table-processing branch, plus
`references/bibliography.json` from the parallel bibliography branch.

## Output

`references/reference_matches.json`.

## Module Contract

See `data-contracts/reference-matches-json.md`.

## Default Implementation

This stage is retained in the legacy thesis workflow but is not yet implemented in the rebuilt `src/tabulus` library.

The target matcher supports numeric references, DOI references, author-year references, author-only fallback, and normalized text containment. It should produce row-level match metadata without mutating the prediction CSV used for table-reconstruction evaluation.

This is the first stage where the table-processing branch and bibliography
branch converge. Bibliography extraction should not be modeled as consuming
MinerU crops or reconstructed table CSVs.

## Verification

The step succeeds when each reference-like table has a detected reference column and row-level match records.
