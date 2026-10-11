# Step 2 Table Reconstruction Quality

Step 2 evaluation scores one reconstructed table prediction CSV against one
curated table `gold.csv`. It measures raw table reconstruction before
reference-table classification, reference matching, scholarly resolution, or
Step 7 resolved export.

The supported metric is Relative Mapping Similarity (RMS), adapted from the
DePlot table-datapoint metric.

## Public Interface

The library API is `tabulus.evaluation.evaluate_table_reconstruction()`. The
CLI is:

```bash
tabulus evaluate-table-reconstruction \
  --gold /path/to/paper/step2_table_reconstruction/tables/<table_id>/gold.csv \
  --prediction /path/to/reconstruction/predictions/<table_id>.csv \
  --metric rms \
  --text-threshold 0.5 \
  --number-threshold 0.1 \
  --out /path/to/evaluation/step2-table-reconstruction.json
```

`--metric` currently supports `rms`. `--out` is optional. If omitted, the command
prints the score and does not write an evaluation artifact.

## Metric Behavior

Tabulus converts each CSV into the flattened table text expected by RMS. The
first row is treated as headers, the first column as row labels, and each cell
is converted into a datapoint keyed by row label plus column header. A leading
`title | ...` row is treated as a title datapoint. CSV cells are normalized for
whitespace, byte-order marks, and ragged rows before comparison.

The implementation compares table datapoints using:

- Average Normalized Levenshtein Similarity for textual keys and values;
- relative numeric comparison for numeric values;
- optimal assignment over datapoint-key similarity;
- precision, recall, and F1 computed from assigned datapoint scores.

The prediction is evaluated in its original orientation and in transposed
orientation. The orientation with the highest RMS F1 is retained.

## Scores And Thresholds

The low-level RMS function returns precision, recall, and F1 on `[0,1]`. The
public Step 2 evaluator reports and serializes the same values on `[0,100]` and
includes:

- `metric`: `rms`
- `metric_name`: `Relative Mapping Similarity`
- `metric_short_name`: `RMS`
- `implementation`: `DePlot`
- `score_scale`: `[0,100]`
- `precision`
- `recall`
- `f1`

When a single summary value is needed, use RMS F1. The implementation selects
orientation by highest F1 and reports RMS precision, recall, and F1 together.

`--text-threshold` controls the text-similarity cutoff and defaults to `0.5`.
`--number-threshold` controls the numeric relative-error cutoff and defaults to
`0.1`.

## TabulusBench Scope

TabulusBench provides Step 2 reconstruction gold for all 252 papers and 605
curated canonical tables under:

```text
<paper_root>/step2_table_reconstruction/tables/<table_id>/gold.csv
```

The prediction side is a Step 2 table-reconstruction prediction CSV. See the
{doc}`../data-contracts/table-prediction-csv` contract for the production CSV
format.

A resolved CSV is not a valid Step 2 prediction input because Step 7 intentionally
adds reference-resolution columns and may filter or merge outputs.
