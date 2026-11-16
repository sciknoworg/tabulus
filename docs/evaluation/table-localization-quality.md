# Step 1 Table Localization Quality

Step 1 evaluation checks whether PDF profiling recovered the annotated
physical table fragments. It scores the canonical crop inventory in
{doc}`../data-contracts/tables-index-json`; it does not score table content or
geometric bounding-box overlap.

## Run the evaluator

```bash
tabulus evaluate-table-localization \
  --gold /path/to/paper/step1_table_localization/gold.json \
  --prediction /path/to/tabulus-output/table-crops/<paper>/tables_index.json \
  --out /path/to/evaluation/step1-table-localization.json
```

The library API is `tabulus.evaluation.evaluate_table_localization()`.
`--out` is optional; without it, the CLI prints the result. Evaluation reads
both inputs without changing them.

| Input | Contents |
| --- | --- |
| `--gold` | One paper's Step 1 `gold.json`, with a `tables` list of physical fragments and page/crop metadata. |
| `--prediction` | The Step 1 `tables_index.json`, with a `tables` list of detected canonical crops. |

## Alignment and scores

The evaluator first matches an unused prediction by a unique identity key
(`table_id`, crop/image name, or annotation key) with compatible page metadata.
Remaining fragments with page metadata can align by order **within the same
page**. That fallback supports controlled manifests whose fragment IDs differ.

The JSON report includes `gold_fragments`, `predicted_fragments`,
`matched_fragments`, false positives/negatives, precision, recall, F1,
`matched_by_identity`, `matched_by_page_order`, and individual `matches`.
Precision, recall, and F1 use `[0,1]` in JSON and percentages in CLI output.
`geometry_metric` is null, with a `geometry_note` explaining the absence of
authoritative gold boxes.

## TabulusBench scope

TabulusBench has `pseudo_gold` for 610 physical fragments across 253 papers.
Its gold records identify fragments and crops but do not supply authoritative
bounding boxes for IoU. Report Step 1 as fragment recovery/alignment, not IoU.
A recovered crop can still yield an empty or incorrect Step 2 reconstruction;
see {doc}`table-extraction-quality` for the separate content metric.
