# Step 2 Table Reconstruction Quality

Step 2 evaluation compares one reconstructed prediction CSV with one curated
`gold.csv`. It scores the **raw physical table**, before reference-table
classification, matching, resolution, or Step 7 enrichment.

Controlled Step 2, explicitly aligned production Step 2, and physical Step 7
use the same scorer for `rms`, `exact-mapping`, and `exact`. See
{doc}`physical-grid-evaluation` for input preparation, batch macro/micro
reporting, CPU workers, checkpoints, resume, and matched subsets.

## Run the evaluator

```bash
tabulus evaluate-table-reconstruction \
  --gold /path/to/paper/step2_table_reconstruction/tables/<table_id>/gold.csv \
  --prediction /path/to/reconstruction/predictions/<table_id>.csv \
  --metric rms \
  --out /path/to/evaluation/step2-table-reconstruction.json
```

The library API is `tabulus.evaluation.evaluate_table_reconstruction()`.
`--out` is optional; without it, the CLI prints the score and writes no report.
The prediction is a Step 2 CSV, as described in the
{doc}`../data-contracts/table-prediction-csv` contract.

| Option | Meaning |
| --- | --- |
| `--metric` | `rms` (default), `exact-mapping`, or `exact`. |
| `--text-threshold` | RMS text-similarity cutoff; default `0.5`. |
| `--number-threshold` | RMS relative numeric-error cutoff; default `0.1`. |

All three public metrics report precision, recall, and F1 on `[0,100]`, but
they define a match differently.

## Metrics

### Relative Mapping Similarity (RMS)

The default RMS metric adapts DePlot's table-datapoint metric. Tabulus
normalizes CSV whitespace, byte-order marks, and ragged rows, then converts
each cell to a mapping keyed by its row label and column header. The first row
supplies headers and the first column supplies row labels; a leading
`title | ...` row becomes a title datapoint.

RMS compares textual keys/values with Average Normalized Levenshtein
Similarity, numerical values with relative error, and datapoints with optimal
assignment. It scores both original and transposed prediction orientations
and retains the one with higher F1. The low-level score uses `[0,1]`; the public
evaluator scales it to `[0,100]`. Its JSON identifies `metric: rms`,
`metric_name: Relative Mapping Similarity`, `metric_short_name: RMS`, and
`implementation: DePlot`. Use **RMS F1** as the primary summary score.

### Exact Mapping Match

`--metric exact-mapping` uses the **same normalized row-label/column-header
mappings** as RMS but requires exact string equality of each key/value pair.
There is no text partial credit or numeric tolerance: `300` and `300.0`
differ. Mapping conversion normalizes whitespace and case. Row/column
reordering preserves mappings, while repeated normalized keys retain their
last value. Blank values count as mappings; a leading title contributes one.

Matches divided by predicted/gold mapping counts give precision/recall and
their harmonic-mean F1. The scorer keeps the higher-F1 result from original
or transposed orientation, using RMS's tie preference. For a single pair:

| Mapping sets | Precision / recall / F1 on `[0,1]` |
| --- | --- |
| Both empty | `1 / 1 / 1` |
| Prediction empty, gold nonempty | `1 / 0 / 0` |
| Gold empty, prediction nonempty | `0 / 1 / 0` |

The public API scales these results to `[0,100]`. The low-level API is
`tabulus.evaluation.exact_mapping_match()`; the CSV API accepts
`metric="exact-mapping"`. Its JSON uses the name **Exact Mapping Match** and
abbreviation **EM**.

### Position-based Exact Cell Match

`--metric exact` compares nonempty values at the **same physical row and
column**, including header cells. It applies Unicode NFC and collapses
whitespace while preserving case, punctuation, digits, and units. It neither
transposes nor shifts the table. Precision/recall divide matches by nonempty
prediction/gold cells; an empty denominator gives zero.

The separate `exact_grid` flag requires identical normalized values, blank
positions, and row widths. The JSON's `exact_cells` object contains those
counts and the flag. RMS thresholds do not affect this metric.

## TabulusBench scope and batch scoring

The expanded benchmark provides 610 curated physical `gold.csv` tables across
253 papers at:

```text
<paper_root>/step2_table_reconstruction/tables/<table_id>/gold.csv
```

The repository's benchmark scorer evaluates **saved controlled predictions**
without rerunning reconstruction models:

```bash
python scripts/evaluation/step2/evaluate_exact_match.py \
  --root "$HOME/tabulusbench" \
  --out "$HOME/tabulus-step2-exact-mapping" \
  --run-id run_01
```

It reads `metadata/tables.csv`, requires 610 distinct gold fragments, and
discovers available adapters. Use `--adapters chandra tesseract-tatr` to limit
the methods. Each method gets per-fragment scores and both macro and micro
summaries for the selected metrics in the same batch report. Missing, empty,
or unusable predictions receive zero fragment scores.
This batch coverage rule overrides the single-pair empty-input convention.

**Macro** averages fragment precision, recall, and F1 separately, giving each
physical table equal weight. **Micro** pools match credit and gold/prediction
counts across fragments, then computes precision, recall, and F1 from those
totals. Missing outputs retain their gold counts in micro recall. RMS and
`exact-mapping` pool normalized mappings; `exact` pools nonempty physical cells.
The two averages can differ without any change to the underlying table scores.
See {doc}`physical-grid-evaluation` for the exact denominators and empty-input
rules.

The shared guide also documents `evaluate-table-batch` and the saved-output
script's other settings.

Reports include paths, hashes, status, scores, method JSON, `summary.json`,
and `summary.csv`; choose a new output directory.

Missing and unusable saved inputs receive zero independently of whether an
older RMS report contains a historical score for that fragment.

When saved MinerU-native results exist at
`results/step1/controlled/evaluation/mineru-native-252-final` (or a supplied
`--native-results` path), the scorer includes them as a **separate baseline**.
It reuses saved fragment correspondence and CSVs; where an older native
manifest lacks a fragment, a saved Step 1 table body can be parsed only after
unique stable fragment-key/page identification. Prediction IDs alone do not
establish correspondence. Derived CSVs go into the report. This
baseline uses MinerU-detected crops, not the controlled adapter inputs.

## Step 7 comparison

A resolved CSV has appended metadata and is not a Step 2 prediction input.
The {doc}`step7-quality` physical evaluator removes each side's declared
metadata columns before scoring the original grid with the same three
metrics. Step 2 Exact Mapping Match
compares normalized mapping identities; Position-based Exact Cell Match and
Step 7 `exact` compare coordinates. Compare Steps 2 and 7 only on matched
fragments with compatible gold grids, the **same metric**, and the same
missing-output policy. Optional Step 7 continuation merging and ORKG
projection are separate outputs.
