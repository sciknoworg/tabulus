# Step 5 Reference Matching Quality

Step 5 evaluation scores the link between a citation cell in a selected
reference table and one or more numbered bibliography positions. The production
artifact is {doc}`../data-contracts/reference-matches-json`, written as
`references/reference_matches.json`.

Step 5 is deterministic and offline. It links table citation occurrences to
bibliography positions; Step 6 performs scholarly identity resolution later.

## Public Interface

The library API is `tabulus.evaluation.evaluate_reference_matching()`. The CLI
is:

```bash
tabulus evaluate-reference-matching \
  --gold /path/to/paper/step5_reference_matching/gold.json \
  --prediction /path/to/reconstruction/references/reference_matches.json \
  --out /path/to/evaluation/step5-reference-matching.json
```

`--out` is optional. The public CLI compares one gold file to one Step 5
prediction artifact. The library API also accepts an optional
`bibliography_index_map` for controlled comparisons where prediction indices
must be mapped to a gold bibliography numbering.

## Gold Schema

Canonical gold lives at:

```text
<paper_root>/step5_reference_matching/gold.json
```

The gold JSON contains a `tables` list. Each table record includes:

- `table_id`
- `table_key`
- `gold_csv`
- `reference_column_index`
- `reference_column_header`
- `citation_cells`

Each `citation_cells[]` record contains:

- `row_index`
- `column_index`
- `value`
- `bibliography_indices`

This gold schema is separate from the production `reference_matches.json`
contract. Production Step 5 uses positions in the normalized Step 4
bibliography order. Controlled evaluation may use gold bibliography numbering
or an explicit index map to isolate Step 5 from Step 4 segmentation changes.

## Alignment Behavior

The evaluator aligns gold tables to predicted `matched_tables` using keys from
`table_id`, `source_file`, `source_prediction`, and `source_parsed`. For each
aligned table, it compares the predicted reference-column index and the citation
cell links.

Citation cells are aligned by citation surface when predictions include cell
values. Numeric citation signatures expand ranges such as `20--22` to the tuple
`20, 21, 22`. Historical controlled artifacts that omit cell surface values are
aligned by table-local match order.

Header rows marked with `is_header: true` are excluded from link scoring.

## Metrics

The serialized result includes:

- `gold_tables`
- `aligned_tables`
- `table_coverage`
- `reference_columns_correct`
- `reference_column_accuracy`
- `gold_citation_cells`
- `surface_matched_cells`
- `citation_surface_recall`
- `exact_link_set_cells`
- `exact_cell_accuracy`
- `gold_links`
- `predicted_links`
- `link_true_positives`
- `link_false_positives`
- `link_false_negatives`
- `link_precision`
- `link_recall`
- `link_f1`

`exact_cell_accuracy` requires the predicted bibliography-index set for an
aligned citation cell to exactly equal the gold set. Individual-link metrics are
computed over `(citation cell, bibliography index)` links.

## TabulusBench Scope

Canonical Step 5 gold currently covers only the P251/P252 deep subset:

- 57 reference-bearing physical fragments
- 2,016 citation cells
- 5,963 bibliography-link occurrences
- gold type: `derived_gold`

This is not the full set of 574 reference-bearing fragments in TabulusBench.
Production Step 5 can be run wherever Step 3 and Step 4 artifacts exist, but
coverage, unmatched-token counts, and consistency diagnostics must not be called
precision, recall, F1, or accuracy unless suitable Step 5 gold labels exist.

The old controlled-run result table with 100% scores is not documented as a
current adapter-level benchmark result. Use the public evaluator against the
current canonical Step 5 gold and the exact prediction artifact being reported.

## Boundary

Step 5 output remains table-cell level. Step 6 consumes one or more Step 5
artifacts, takes the paper-level union of linked bibliography indices, and
resolves each unique bibliography entry once. Step 5 does not resolve DOIs,
mutate prediction CSVs, or establish scholarly identity correctness.
