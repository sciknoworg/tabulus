# Step 8: Reference Matching

## Goal

Match table reference cells to bibliography entries.

## Input

Reference-like tables and `references/bibliography.json`.

## Output

`references/reference_matches.json`.

## Module Contract

See `data-contracts/reference-matches-json.md`.

## Default Implementation

The current matcher supports numeric references, DOI references, author-year references, author-only fallback, and normalized text containment.

## Verification

The step succeeds when each reference-like table has a detected reference column and row-level match records.
