# Step 8: Reference Matching

## Goal

Match table reference cells to bibliography entries.

## Input

Reference-like normalized tables or prediction CSV content, plus `references/bibliography.json`.

## Output

`references/reference_matches.json`.

## Module Contract

See `data-contracts/reference-matches-json.md`.

## Default Implementation

This stage is retained in the legacy thesis workflow but is not yet implemented in the rebuilt `src/tabulus` library.

The target matcher supports numeric references, DOI references, author-year references, author-only fallback, and normalized text containment. It should produce row-level match metadata without mutating the prediction CSV used for table-reconstruction evaluation.

## Verification

The step succeeds when each reference-like table has a detected reference column and row-level match records.
