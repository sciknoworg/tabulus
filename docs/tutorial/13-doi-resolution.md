# Planned Stage: DOI Resolution

## Goal

Resolve DOI values for matched bibliography entries.

## Input

Reference matches and bibliography entries.

## Output

Reference matches enriched with DOI values.

## Default Implementation

This stage is retained in the legacy thesis workflow but is not yet implemented in the rebuilt `src/tabulus` library.

The target resolver first uses DOI values parsed from bibliography text. If enabled, it can query Crossref for missing DOI values. DOI enrichment happens after table reconstruction and reference matching; it should not overwrite the prediction CSV artifact used for table-quality evaluation.

## Verification

The step succeeds when DOI values are attached where available and unresolved rows remain traceable.
