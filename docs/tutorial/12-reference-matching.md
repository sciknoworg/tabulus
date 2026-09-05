# Step 5: Reference Matching

## Goal

Stage 5 links reference cells in Stage 3-selected reconstructed tables to
entries in the Stage 4 bibliography artifact.

This is where the table-processing branch and bibliography branch converge:

```text
Stage 3: selected_reference_tables.json
          \
           \
            -> Stage 5 reference matching
           /
          /
Stage 4: references/bibliography.json

            |
            v

references/reference_matches.json
```

## Input

Stage 5 requires two files:

1. `selected_reference_tables.json`
   : The Stage 3 selection manifest. It identifies tables classified as
     reference-like and points to their existing reconstruction artifacts.

2. `references/bibliography.json`
   : The Stage 4 bibliography artifact extracted from the original PDF.

Stage 5 does not consume the original PDF, rerun table reconstruction, or call
GROBID. It reads the selected-table manifest and the already-created
bibliography JSON.

## Command Line

Run reference matching with:

```bash
tabulus match-references \
  --selected /path/to/selected_reference_tables.json \
  --bibliography /path/to/references/bibliography.json
```

To choose the output file explicitly:

```bash
tabulus match-references \
  --selected /path/to/selected_reference_tables.json \
  --bibliography /path/to/references/bibliography.json \
  --out /path/to/reference_matches.json
```

If `--out` is omitted, Tabulus writes:

```text
<reconstruction-directory>/references/reference_matches.json
```

## Output

`reference_matches.json` records:

- selected, checked, and skipped reference-table counts
- detected reference column for each checked table
- row-level reference-cell matches
- matched bibliography indices
- match method provenance
- unmatched tokens where applicable
- skipped-table diagnostics when a parsed-table artifact cannot be used safely

See {doc}`../data-contracts/reference-matches-json` for the full artifact
schema.

## Matching Behavior

Stage 5 is deterministic and offline. It does not query Crossref, GROBID, an
LLM, embeddings, external search, or any metadata service.

The matcher records these method labels:

- `numeric_position`
- `doi_exact`
- `author_year`
- `author_only`
- `text_containment`

`numeric_position` interprets numeric table references as one-based positions
in the normalized GROBID TEI bibliography order stored in
`references/bibliography.json`. For example, `[12]` links to bibliography entry
12. This is positional linkage, not DOI enrichment.

Numeric normalization handles common lists and ranges, such as `[12, 14]`,
`[12-15]`, `88 and 89`, and `83, 90, and 91`. It also recovers conservative
OCR-spacing cases inside numeric-only cells. Textual author-year forms such as
`Smith (2020)` are not treated as numeric references.

Author-based matching uses conservative normalized author/year,
author-only, and text-containment fallbacks. Ambiguous textual matches may keep
multiple candidate bibliography entries rather than silently choosing one.

## Skipped Tables

Stage 5 skips a selected table instead of guessing when the referenced parsed
artifact contains:

- `no_parsed_table`
- `multiple_parsed_tables`

The skipped table is recorded in `skipped_tables`, and the rest of the matching
run can continue. Malformed input contracts or identity mismatches are treated
as errors.

## Boundary To Stage 6

Stage 5 links table references to bibliography entries. External DOI lookup and
identifier enrichment belong to Stage 6.

Stage 5 does not mutate raw reconstruction prediction CSVs.
