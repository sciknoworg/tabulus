# Resolved CSV

Step 7 exports user-facing bibliography-resolved CSV files after
deterministic reference matching and paper-level scholarly reference
resolution.

A prediction CSV and a resolved CSV are deliberately different artifacts:

- **prediction CSV:** the Step 2 reconstruction of one physical table crop,
  before bibliography enrichment; this remains the artifact used for
  table-reconstruction evaluation
- **resolved CSV:** a Step 7 derivative containing the original physical
  table rows plus bibliography-resolution enrichment

Step 7 never rewrites Step 2 prediction CSV files.

## Inputs

Step 7 consumes:

1. one Step 5 `references/reference_matches.json` artifact
2. one Step 6 `references/reference_resolution.json` registry

Optional continuation merging additionally consumes the Step 1
`tables_index.json` continuation topology.

Step 7 performs no Crossref, CORE, LLM, DOI, or bibliographic search. It is
a deterministic join over previously persisted artifacts.

## Default physical export

```bash
tabulus export-resolved-csv \
  --reference-matches /path/to/reconstruction/references/reference_matches.json \
  --reference-resolution /path/to/artifacts/references/reference_resolution.json
```

By default Tabulus writes:

```text
<reconstruction>/
  resolved_reference_tables/
    <prediction-stem>_resolved.csv
    resolved_tables.json
```

Every exported CSV retains the original physical-table rows, row order,
reconstructed citation/reference values, and provenance to the Step 2
prediction CSV.

## Enrichment columns

Step 7 appends:

```text
tabulus_reference_indices
tabulus_resolution_statuses
tabulus_canonical_dois
tabulus_canonical_titles
tabulus_canonical_authors
tabulus_canonical_years
tabulus_canonical_venues
tabulus_resolution_sources
tabulus_resolution_confidences
tabulus_resolution_reasons
tabulus_raw_references
tabulus_unmatched_tokens
```

Each enrichment cell is a compact JSON array. This preserves positional
correspondence when one table cell cites multiple bibliography entries.

Rejected Step 6 identities and Step 5 unmatched tokens remain explicit.

## Header handling

When Step 5 identifies a physical header row, the appended cells on that row
contain the enrichment-column names. When no header is identifiable, Step 7
does not invent one, so physical row indices are not shifted.

## Optional continuation merging

Physical tables remain the default Step 7 outputs. Logical merging is
opt-in:

```bash
tabulus export-resolved-csv \
  --reference-matches /path/to/reference_matches.json \
  --reference-resolution /path/to/reference_resolution.json \
  --merge-continuations
```

Step 7 uses the explicit Step 1 continuation topology and revalidates
reconstructed-table compatibility at merge time. Safe mappings are exact
repeated headers with equal column count, or same-width positional
continuation when no repeated header is present.

Incompatible or incomplete groups remain separate. Physical resolved CSVs
are always retained.

Successful merged tables are additional artifacts under:

```text
resolved_reference_tables/merged/
```

Merged CSVs append:

```text
tabulus_physical_table_id
```

so every merged row remains traceable to its physical source table.

## Scientific boundary

Continuation merging is not part of Step 2 reconstruction evaluation.

```text
physical prediction CSV
↔
physical gold CSV
```

remains the Step 2 evaluation unit.
