# Step 6: Reference-Table Classification

## Goal

Decide whether an extracted table contains citations or reference identifiers.

## Input

Normalized OCR tables.

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

The current backend uses regex-based header and citation detection.

## Verification

The step succeeds when every OCR table has an explicit classification decision and evidence.
