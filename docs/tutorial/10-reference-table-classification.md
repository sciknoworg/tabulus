# Step 3: Reference-Table Classification

## Goal

Decide which reconstructed physical tables contain reference-like scientific citation content and should later enter the reference-resolution branch.

This stage is implemented in the rebuilt library as:

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
```

The manifest records a routing/classification decision for each physical table considered. It does not overwrite:

- `native/`
- `parsed/`
- `predictions/`
- `batch_summary.json`

A non-reference classification means only that the table should not proceed down the future reference-resolution branch. It does not mean the reconstruction is invalid.

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

Every physical table is classified independently first. The classifier uses the common parsed rows produced during reconstruction, preserves the legacy reference-bearing table heuristics, and records matched evidence.

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
physical table
  -> independent reference classification
  -> continuation relationship resolution
  -> final reference-table decision
```

An explicitly identified continuation may inherit a positive reference-table classification from its preceding logical table. The manifest preserves whether the final decision came from independent table evidence or continuation inheritance.

This does not merge files. Continued tables remain separate physical entities through MinerU detection, canonical crops, reconstruction, parsed artifacts, prediction CSVs, and classification.

## Boundary

This stage performs reference-table routing only. It does not extract bibliographies, match references, resolve DOI values, write resolved CSVs, merge continued tables, or run the complete end-to-end pipeline.

## Next Step

The next rebuilt stages are planned bibliography extraction, reference
matching, DOI resolution, and resolved CSV export. Bibliography extraction is
a parallel PDF-level branch that should produce `references/bibliography.json`
from the original PDF before converging with classified reference-like tables
at reference matching.
