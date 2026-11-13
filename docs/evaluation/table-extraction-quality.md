# Step 2 Table Reconstruction Quality

Step 2 evaluation scores one reconstructed table prediction CSV against one
curated table `gold.csv`. It measures raw table reconstruction before
reference-table classification, reference matching, scholarly resolution, or
Step 7 resolved export.

The default metric is Relative Mapping Similarity (RMS), adapted from the
DePlot table-datapoint metric. Exact Mapping Match measures strict equality
using the same mapping representation. The earlier Exact Cell Match metric
also remains available for position-based comparisons.

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

`--metric` supports `rms`, `exact-mapping`, and `exact`. `--out` is optional. If omitted, the command
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

## Exact Mapping Match

Use `--metric exact-mapping` for exact-match precision, recall and F1 with
the same CSV normalization and row-label/column-header keys as RMS.
Each equal normalized key/value pair earns one match. Precision divides
matches by predicted mappings, recall divides matches by gold mappings,
and F1 is their harmonic mean. Values use string equality, so `300` and
`300.0` differ. Whitespace is normalized and case is lowered using the
existing RMS conversion. No numeric tolerance or partial credit applies.

The evaluator scores the original and transposed prediction and keeps the
orientation with higher F1. Equal F1 retains the same orientation preference
as RMS. Reordering rows or columns preserves the mapping identities.
Repeated normalized keys retain their last value, following the RMS
dictionary representation. Blank values count as mappings, and a leading
`title | ...` line contributes a title mapping.

Empty-mapping behavior also follows RMS. Two empty mapping sets score
`P=R=F1=1`. Empty prediction mappings against nonempty gold score
`P=1, R=0, F1=0`. Empty gold mappings against nonempty prediction score
`P=0, R=1, F1=0`. The public API scales these values to `[0,100]`.

The low-level API is `tabulus.evaluation.exact_mapping_match()`.
The normal CSV entry point is
`evaluate_table_reconstruction(gold, prediction, metric="exact-mapping")`.
Its JSON identifies the metric as `exact-mapping`, with name
`Exact Mapping Match` and abbreviation `EM`. Existing RMS artifacts and
the earlier `exact` metric retain their meaning.

## Position-based Exact Cell Match

With `--metric exact`, a nonempty cell earns one match when its value equals
the gold value at the same row and column. The scorer uses Unicode NFC and
collapses whitespace. It preserves case, punctuation, digits, and units.
Headers are scored as cells. The scorer uses the supplied orientation and
positions, so a shifted row or column can reduce the score.

Precision divides matches by nonempty predicted cells. Recall divides matches
by nonempty gold cells. F1 is their harmonic mean. Scores use the same
`[0,100]` scale as RMS. An empty denominator gives zero. Empty cells contribute
no cell matches, but the separate `exact_grid` flag requires the same normalized
values, row widths, and blank positions. The JSON's `exact_cells` object contains
the counts and that flag. RMS thresholds apply only to RMS.

For a benchmark comparison, score both metrics on the same table pairs and
use the same aggregation and missing-prediction policy. To compare Step 2
with a physical Step 7 export, use the original table grid from each output.
Remove the appended metadata columns from the Step 7 gold and prediction
separately, then call `tabulus.evaluation.count_exact_cells()` on those grids.
The `evaluate-step7-physical` command performs this extraction and reports
both exact and RMS scores. See the {doc}`../data-contracts/resolved-csv`
contract for its manifest and alignment inputs.

## TabulusBench Scope

The expanded TabulusBench set provides 610 controlled gold fragments across
253 papers, including the original 605-fragment population, under:

```text
<paper_root>/step2_table_reconstruction/tables/<table_id>/gold.csv
```

The prediction side is a Step 2 table-reconstruction prediction CSV. See the
{doc}`../data-contracts/table-prediction-csv` contract for the production CSV
format.

A resolved CSV is not a valid Step 2 prediction input because Step 7 intentionally
adds reference-resolution columns and may filter or merge outputs.

### Scoring saved controlled predictions

Run the benchmark scorer from the software checkout with Tabulus installed.

```bash
python scripts/evaluation/step2/evaluate_exact_match.py \
  --root "$HOME/tabulusbench" \
  --out "$HOME/tabulus-step2-exact-mapping" \
  --run-id run_01
```

The scorer reads `metadata/tables.csv`, requires 610 distinct gold fragments,
and discovers the adapters in the saved controlled results. Use
`--adapters chandra tesseract-tatr` to select methods explicitly.
Every method gets per-fragment results and separate macro averages of P,
R and F1 over all 610 gold fragments. Missing, empty, or unusable predictions
receive zero for all three metrics in this benchmark aggregation.
This coverage policy overrides the single-pair empty-input convention.
Existing archived failure statuses also receive zero.

Outputs include a JSON report for each method, `summary.json`, and
`summary.csv`. Each fragment records its source paths, hashes, status and
scores. Use a new output directory. No model inference is performed.

The scorer includes the MinerU-native baseline when saved native results
are available at
`results/step1/controlled/evaluation/mineru-native-252-final`, or a directory
supplied through `--native-results`. It reuses that evaluation's saved
fragment correspondence and CSVs. For fragments absent from the older native
manifest, it can parse a saved Step 1 table body only when table identity
and page give a unique match. Those derived CSVs are saved with the report.
Native output uses MinerU's own detected crops, so report it as a separate
baseline alongside the controlled adapters.
