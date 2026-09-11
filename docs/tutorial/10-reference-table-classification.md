# Step 3: Reference-Table Classification

## Goal

Decide which reconstructed-table instances contain reference-like scientific
citation content and should enter the reference-processing branch.

This step is implemented in the rebuilt library as:

```bash
tabulus classify-reference-tables
```

## Input

Reference-table classification consumes reconstruction artifacts from one adapter:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      parsed/
      predictions/
      batch_summary.json
```

The classifier reads the common parsed table representation and the reconstruction batch manifest. It does not read the original PDF, rerun OCR, or modify prediction CSVs.

## Output

By default, the command writes:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      reference_table_classification.json
      selected_reference_tables.json
```

`reference_table_classification.json` records a routing/classification decision
for each reconstructed-table instance considered. `selected_reference_tables.json` is a
non-destructive pointer manifest containing only the tables selected for Step
5. Neither artifact overwrites:

- `native/`
- `parsed/`
- `predictions/`
- `batch_summary.json`

A non-reference classification means only that the table should not proceed
down the reference-processing branch. It does not mean the reconstruction is
invalid.

## CLI

Classify one reconstruction directory:

```bash
tabulus classify-reference-tables \
  --reconstruction "/path/to/table-crops/<paper>/reconstructions/<adapter>"
```

Classify all immediate crop roots beneath a table-crops parent for one adapter:

```bash
tabulus classify-reference-tables \
  --crops-folder "/path/to/tabulus-output/table-crops" \
  --adapter paddleocr-vl
```

Classify reconstruction directories listed in a UTF-8 text file:

```bash
tabulus classify-reference-tables \
  --reconstruction-list "/path/to/reconstructions.txt"
```

For multi-paper classification, the default manifest is written inside each selected reconstruction directory. `--out` is only valid when exactly one reconstruction directory is selected.

## Classification Model

One deterministic regex/rule classifier is applied independently to the outputs
of each table-reconstruction method. Every reconstructed-table instance is
classified independently first. The classifier uses the common parsed rows produced during reconstruction, preserves the legacy reference-bearing table heuristics, and records matched evidence.

The manifest includes fields such as:

- `is_reference_table`
- `independent_is_reference_table`
- `classification_source`
- `continued_from_table_id`
- `continuation_caption`
- `matched_header_cells`
- `matched_citation_cells`
- `reason`

Current heuristics include reference-like headers, citation-like cell content, DOI-like strings, author-year patterns, and conservative bare numeric references when those numbers occur inside explicitly reference-like columns such as `Refs.`, `References`, or `Citations`.

## Continued Tables

Continued-table handling is a separate layer on top of independent classification:

```text
reconstructed-table instance
  -> independent reference classification
  -> continuation relationship resolution
  -> final reference-table decision
```

An explicitly identified continuation may inherit a positive reference-table classification from its preceding logical table. The manifest preserves whether the final decision came from independent table evidence or continuation inheritance.

This does not merge files. Continued-table fragments retain separate crops and reconstruction artifacts
through parsing, prediction CSV export, and classification.

## Boundary

This step performs reference-table routing only. It does not extract bibliographies, match references, resolve DOI values, write resolved CSVs, merge continued tables, or run the complete end-to-end pipeline.

## Next Step

The next rebuilt branch is bibliography extraction, which produces
`references/bibliography.json` from the original PDF. It runs in parallel with
table processing and converges with classified reference-like tables at Step 5
reference matching. Step 6 resolves the union of referenced
bibliography indices once per paper; Step 7 exports resolved CSVs from Step 5
matches and the Step 6 paper-level registry.

Without human gold-standard labels, evaluate Step 3 coverage, consistency,
and agreement across reconstruction outputs rather than accuracy.
