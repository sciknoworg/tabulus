# Step 9: DOI Resolution

## Goal

Resolve DOI values for matched bibliography entries.

## Input

Reference matches and bibliography entries.

## Output

Reference matches enriched with DOI values.

## Default Implementation

The current matcher first uses DOI values parsed from bibliography text. If enabled, it queries Crossref for missing DOI values.

## Verification

The step succeeds when DOI values are attached where available and unresolved rows remain traceable.
