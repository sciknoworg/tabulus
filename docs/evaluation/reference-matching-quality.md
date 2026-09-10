# Reference Matching Quality

Reference matching quality concerns Step 5 table-cell-to-bibliography-position
links. Step 5 is deterministic and offline: it links citations found in
selected reference-table cells to ordered Step 4 bibliography entries. DOI
values present in `references/bibliography.json` are extraction evidence, not
independently resolved scholarly identities.

The production Step 5 command is:

```bash
tabulus match-references \
  --selected /path/to/selected_reference_tables.json \
  --bibliography /path/to/artifact-root/references/bibliography.json
```

The output is `references/reference_matches.json` beside the reconstruction
adapter directory unless `--out` is provided.

## Current Implementation Status

Tabulus does not currently expose a library-native evaluator or `tabulus`
evaluation command for Step 5 reference matching quality. The production
artifact includes coverage and traceability counts, but those counts are not a
human-validated matching score by themselves.

## Scored Artifact

A Step 5 evaluator would score `references/reference_matches.json` against
curated labels for table-cell or token links. The labels must specify which
table occurrences should link to which bibliography positions.

Metrics should distinguish:

- detecting the reference-like cell or token in the reconstructed table;
- linking that occurrence to the intended Step 4 bibliography entry;
- preserving unmatched tokens and skipped tables for audit;
- carrying through Step 4 DOI fields when present as extraction evidence.

Without human gold-standard labels, report coverage, consistency, agreement,
and unmatched-token or unmatched-row counts. Precision, recall, F1, and
accuracy require suitable curated labels; successful links alone do not
establish correctness.

## Step 6 Diagnostic Boundary

Step 6 scholarly reference resolution uses a paper-level denominator: the
union of bibliography indices linked by applicable Step 5 artifacts,
deduplicated by bibliography index. Each unique bibliography entry is resolved
once per paper, not once per table cell, table fragment, or reconstruction
adapter.

Resolution status counts and resolution coverage are diagnostics over that
paper-level denominator. They should not be presented as accuracy without a
human gold standard. Operational failures during resolution are distinct from
scientific rejections and should not be counted as rejected scholarly
references.

Planned Step 7 export will join Step 6 paper-level identities back to every
relevant table occurrence. Step 7 is not currently implemented and should not
be used as a table reconstruction, matching, or scholarly-resolution evaluator.
