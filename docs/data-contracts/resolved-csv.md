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

The optional ORKG projection uses a separate structural plan and output
manifest. See {doc}`step7-structure`.

## Inputs

Step 7 consumes:

1. one Step 5 `references/reference_matches.json` artifact
2. one Step 6 `references/reference_resolution.json` registry

Optional continuation merging additionally consumes the Step 1
`tables_index.json` continuation topology.

The physical export is a deterministic join over previously persisted
artifacts. The optional ORKG projection can use an LLM for structural
planning and takes scholarly identities from the same Step 6 registry.

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

Evaluate the original physical grids with both Exact Cell Match and RMS.
The command reads the gold and prediction manifests and removes each
side's declared trailing metadata columns before scoring.

```powershell
tabulus evaluate-step7-physical --prediction resolved_tables.json --benchmark-paper C:\Datasets\tabulusbench\path\P251 --alignment alignment.json --out physical-scores.json
```

Alternatively, supply the physical gold manifest with `--gold`.
`--metric` accepts `both` (the default), `exact`, or `rms`. RMS requires
the optional evaluation dependencies installed with
`python -m pip install "tabulus[evaluation]"`.

The alignment JSON supplies explicit fragment correspondence.

```json
{
  "schema_version": 1,
  "tables": [
    {"table_id": 20, "table_key": "page_026_table_018"}
  ]
}
```

Here `table_id` identifies a prediction-manifest entry and `table_key`
identifies a gold-manifest entry. Without `--alignment`, each prediction
entry must supply its own stable `table_key`. Numeric IDs alone do not
establish correspondence. Paths inside manifests are resolved relative to
their manifest directory unless absolute.

The report includes per-fragment precision, recall and F1, macro averages
over all gold fragments, exact-grid matches, export coverage and input
hashes. Missing or empty exports contribute zero to all three scores.
Extra or multiply assigned prediction fragments raise an error.
The evaluator checks declared metadata headers where a header row is
available. Exact values use Unicode NFC and collapsed whitespace, preserving
case and punctuation. RMS uses the Step 2 normalization, default thresholds
and orientation selection. Both scores include original headers and citation
text. Citation associations and scholarly identities are evaluated separately.

Place the report outside the gold directory. Evaluation output cannot
overwrite an input artifact.

Continuation merging is not part of Step 2 reconstruction evaluation.

```text
physical prediction CSV
↔
physical gold CSV
```

remains the Step 2 evaluation unit.
