# Step 6: Reference-Table Classification

## Goal

Decide whether an extracted table contains citations or reference identifiers.

## Input

In the rebuilt pipeline, this stage should be downstream of table
reconstruction. The intended inputs are the common parsed representation under
`reconstructions/<adapter>/parsed/` and the reconstruction batch manifest
`reconstructions/<adapter>/batch_summary.json`.

## Output

When rebuilt, the stage should write a classification manifest beside the
reconstruction artifacts:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      reference_table_classification.json
```

It should not overwrite `native/`, `parsed/`, `predictions/`, or
`batch_summary.json`.

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

A non-reference classification should mean only that the table does not proceed
down the reference-resolution branch. It should not mean the reconstruction is
invalid, and it should not cause `predictions/*.csv` files to be deleted.

Continued-table handling should remain a layer on top of independent physical
table classification. Current reconstruction keeps continued table segments as
separate physical table IDs and does not merge their files.

## Verification

The step succeeds when every OCR table has an explicit classification decision and evidence.
