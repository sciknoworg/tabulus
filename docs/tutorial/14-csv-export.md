# Step 7: Resolved CSV Export

## Goal

Join Step 6 paper-level scholarly identities back onto the physical table
rows linked in Step 5 and export final user-facing CSV files.

Step 7 is deterministic. It performs no new scholarly search or model
inference.

## Inputs

```text
<reconstruction>/references/reference_matches.json
<artifact-root>/references/reference_resolution.json
```

## Physical resolved CSV export

```bash
tabulus export-resolved-csv \
  --reference-matches \
  "$RECONSTRUCTION/references/reference_matches.json" \
  --reference-resolution \
  "$ARTIFACT_ROOT/references/reference_resolution.json"
```

By default:

```text
<reconstruction>/
  resolved_reference_tables/
    <prediction-stem>_resolved.csv
    resolved_tables.json
```

Prediction CSVs remain unchanged. Step 7 preserves physical rows and appends
resolution metadata using JSON-array cells, so multi-reference citation
cells remain losslessly aligned.

## Continued tables

Continued physical tables are not merged by default.

```bash
tabulus export-resolved-csv \
  --reference-matches /path/to/reference_matches.json \
  --reference-resolution /path/to/reference_resolution.json \
  --merge-continuations
```

In the canonical layout, Tabulus infers `tables_index.json`. For a
non-standard layout, add:

```text
--tables-index /path/to/tables_index.json
```

The merge uses explicit Step 1 continuation relationships and rechecks
reconstructed-table compatibility. Incompatible or incomplete groups remain
as separate physical resolved CSVs, with the reason recorded in
`resolved_tables.json`.

Successful logical merges are additional files beneath:

```text
resolved_reference_tables/merged/
```

Each merged row retains `tabulus_physical_table_id`.

## Verification

Verify that:

1. physical resolved CSV row counts match their Step 2 predictions
2. Step 2 prediction CSVs are unchanged
3. every Step 5 linked bibliography index has a final Step 6 identity
4. rejected resolutions remain represented
5. multi-reference cells preserve positional JSON-array alignment
6. continuation merging never removes physical resolved files
7. merged rows retain their physical table provenance

## Output contract

See {doc}`../data-contracts/resolved-csv`.
