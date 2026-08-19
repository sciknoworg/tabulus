# Step 6: Reference-Table Classification

## Goal

Decide whether an extracted table contains citations or reference identifiers.

## Input

Normalized table reconstructions or prediction CSV content.

## Output

`tables/reference_table_classification.json` or enriched table objects.

## Module Contract

```json
{
  "table_id": 1,
  "is_reference_table": true,
  "has_tag_match": true,
  "has_citation_match": true,
  "matched_header_cells": ["References"],
  "matched_citation_cells": ["[1]", "Smith et al. 2020"],
  "reason": "Header-like reference tags and citation-like cell content found."
}
```

## Default Implementation

This stage is retained in the legacy thesis workflow but is not yet implemented in the rebuilt `src/tabulus` library.

The target classifier should use evidence such as reference-like headers, bracketed numeric citations, DOI strings, author-year citations, or similar patterns to decide whether a table should enter the reference-resolution branch. It should not change the table prediction CSV used for reconstruction evaluation.

## Verification

The step succeeds when every OCR table has an explicit classification decision and evidence.
