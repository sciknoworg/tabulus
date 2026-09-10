# Table Reconstruction Quality

Table reconstruction quality evaluates whether a reconstructed table prediction
CSV matches a manually curated ground-truth table CSV. It scores raw table
structure and content before reference matching, scholarly resolution, or
Step 7 resolved export.

The current library-native evaluator supports one metric: Relative Mapping
Similarity (RMS), adapted from the DePlot table-datapoint metric.

```text
normalized reconstruction
        |
        v
prediction CSV
        |
        v
Relative Mapping Similarity (RMS)
        |
        v
ground-truth CSV
```

A DOI-enriched or otherwise resolved CSV is not used for table reconstruction
quality because enrichment intentionally changes reference-cell values.

## Current Implementation Status

The public API is `tabulus.evaluation.evaluate_table_reconstruction()`. The
public CLI is:

```bash
tabulus evaluate-table-reconstruction \
  --gold /path/to/gold.csv \
  --prediction /path/to/prediction.csv \
  --metric rms \
  --text-threshold 0.5 \
  --number-threshold 0.1 \
  --out /path/to/evaluation.json
```

`--out` is optional. When it is omitted, the command prints the result and does
not write an evaluation artifact.

RMS uses the optional evaluation dependencies, including SciPy for optimal
assignment. Install them with:

```bash
python -m pip install -e ".[evaluation]"
```

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
- precision, recall, and F1 computed from the assigned datapoint scores.

The prediction is evaluated in its original orientation and in transposed
orientation. The orientation with the highest RMS F1 is retained.

## Scores And Thresholds

The low-level RMS function returns precision, recall, and F1 on `[0,1]`. The
public table-reconstruction API and CLI report the same values on `[0,100]` and
serialize:

- `metric`: `rms`
- `metric_name`: `Relative Mapping Similarity`
- `metric_short_name`: `RMS`
- `implementation`: `DePlot`
- `score_scale`: `[0,100]`
- `precision`
- `recall`
- `f1`

When a single summary value is needed, use RMS F1, because the implementation
selects transposed versus non-transposed orientation by highest F1 and reports
RMS precision, recall, and F1 together.

`--text-threshold` controls the text-similarity cutoff and defaults to `0.5`.
`--number-threshold` controls the numeric relative-error cutoff and defaults to
`0.1`. Both thresholds must be values between `0` and `1`.

## Boundary

Table reconstruction evaluation does not measure reference-table
classification, bibliography extraction, table-cell-to-bibliography matching,
Step 6 scholarly reference resolution, or downstream resolved exports.

Adapter comparisons should export every candidate through the same prediction
CSV contract before scoring. Runtime, hardware, and model-environment context
can be recorded beside the metric, but those measurements are separate from
RMS and should not be called reconstruction accuracy.
