# Step 5: Reference Matching

## Goal

Step 5 links reference cells in Step 3-selected reconstructed tables to
positions in the Step 4 bibliography artifact. This is deterministic
table-cell-to-bibliography-position matching.

This is where the table-processing branch and bibliography branch converge:

```text
Step 3: selected_reference_tables.json
          \
           \
            -> Step 5 reference matching
           /
          /
Step 4: references/bibliography.json

            |
            v

references/reference_matches.json
```

## Input

Step 5 requires two files:

1. `selected_reference_tables.json`
   : The Step 3 selection manifest. It identifies tables classified as
     reference-like and points to their existing reconstruction artifacts.

2. `references/bibliography.json`
   : The Step 4 bibliography artifact extracted from the original PDF.

Step 5 does not consume the original PDF, rerun table reconstruction, or call
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

Step 5 is deterministic and offline. It does not query Crossref, GROBID, an
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

## Aggregation Boundary

Step 5 output is table-cell level. Multiple cells, selected tables, and
reconstruction adapters may refer to the same bibliography index.

Step 6 collects matched bibliography indices across supplied reconstruction
methods for a paper, takes their union, and deduplicates by bibliography index.
The resolution key is:

```text
(paper, bibliography_index)
```

Resolving each unique bibliography entry once is useful because scholarly
identity is a property of the paper-level bibliography entry, not of a
particular reconstructed table cell. It also keeps different reconstruction
adapters from receiving different downstream identity decisions for the same
bibliography entry.

## Skipped Tables

Step 5 skips a selected table instead of guessing when the referenced parsed
artifact contains:

- `no_parsed_table`
- `multiple_parsed_tables`

The skipped table is recorded in `skipped_tables`, and the rest of the matching
run can continue. Malformed input contracts or identity mismatches are treated
as errors.

## Boundary To Step 6

Step 5 links table references to bibliography entries. External DOI lookup and
scholarly-identity resolution are outside Step 5; they belong to Step 6
paper-level reference resolution.

Step 5 does not mutate raw reconstruction prediction CSVs or the Step 4
bibliography evidence. Coverage and agreement measures are documented in
{doc}`../evaluation/reference-matching-quality`; they should not be described
as accuracy without human gold-standard labels.
