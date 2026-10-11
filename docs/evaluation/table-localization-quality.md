# Step 1 Table Localization Quality

Step 1 evaluation scores physical table-fragment recovery from a PDF profiling
run. The production artifact is {doc}`../data-contracts/tables-index-json`,
which records detected table crops and provenance for downstream reconstruction.

Canonical Step 1 gold contains physical-fragment identity and page/crop
metadata, but it does not contain authoritative bounding-box coordinates for IoU
scoring. The current evaluator therefore measures fragment recovery/alignment,
not geometric IoU.

## Public Interface

The library API is `tabulus.evaluation.evaluate_table_localization()`. The CLI
is:

```bash
tabulus evaluate-table-localization \
  --gold /path/to/paper/step1_table_localization/gold.json \
  --prediction /path/to/paper/tabulus_runs/step1/tables_index.json \
  --out /path/to/evaluation/step1-table-localization.json
```

`--out` is optional. When omitted, the command prints the result without writing
an evaluation JSON file.

## Inputs

`--gold`
: Canonical Step 1 `gold.json` for one paper. The evaluator expects a JSON
  object with a `tables` list.

`--prediction`
: A Step 1 production `tables_index.json` artifact. The evaluator expects a
  JSON object with a `tables` list.

Evaluation is read-only with respect to both files.

## Alignment Behavior

The evaluator first tries identity alignment. Gold keys are collected from
fields such as `table_id`, `crop`, and annotation image/table keys. Prediction
keys are collected from `table_id`, `image`, and `image_name`. A prediction is
matched by identity only when the key points to exactly one unused prediction
and the page metadata is compatible.

Unmatched tables with page metadata are then aligned by page order within the
same page. This fallback is intended for controlled manifests that preserve
physical-fragment order but do not share exact table identifiers.

The serialized result includes:

- `gold_fragments`
- `predicted_fragments`
- `matched_fragments`
- `false_positives`
- `false_negatives`
- `precision`
- `recall`
- `f1`
- `matched_by_identity`
- `matched_by_page_order`
- `matches`
- `geometry_metric: null`
- `geometry_note`

Precision, recall, and F1 are reported on `[0,1]` in JSON and printed as
percentages by the CLI.

## TabulusBench Scope

TabulusBench provides Step 1 gold for all 252 papers and 605 physical table
fragments/crops. The gold type is `pseudo_gold`. Because the gold does not
provide authoritative bounding boxes, Step 1 quality must be reported as
fragment recovery/alignment rather than IoU.

Step 1 evaluation is separate from Step 2 table reconstruction. A localized
crop can be recovered by Step 1 even if a later reconstruction adapter returns
an empty or erroneous table.
