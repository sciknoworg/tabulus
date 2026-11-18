# Shared physical-grid evaluation

Controlled Step 2, production Step 2 and physical Step 7 use one scoring
implementation. Their input adapters prepare comparable grids against canonical
Step 2 gold. Citation association, scholarly identity recovery and ORKG
structural projections have separate evaluation targets.

**Macro and micro summarize the same fragment comparisons**, returned
together by batch evaluation. They are not separate metrics or CLI endpoints.
Equivalent prepared gold/prediction grids with the same status and metric
settings receive identical scores, regardless of which Step produced the CSV.

## Input preparation

The canonical physical target is
`step2_table_reconstruction/tables/<table_key>/gold.csv`.
Keep its original headers, rows, blank positions and citation text.

- Controlled Step 2 supplies the saved reconstruction for the canonical crop.
- Production Step 2 supplies an explicitly aligned production reconstruction.
- Physical Step 7 supplies an explicitly aligned resolved CSV. Its declared
  appended metadata columns are removed before scoring.
- The existing enriched Step 7 gold-manifest interface remains supported.
  Its declared metadata columns are removed as well; raw Step 2 gold has none.

Prediction table IDs never establish correspondence with gold IDs by themselves.
Use stable paper/fragment keys and explicit correspondence. Neither the shared
scorer nor its micro aggregator matches content across different table pairs.

## Shared interfaces

| Interface | Input and preparation | Output |
| --- | --- | --- |
| `evaluate_table_reconstruction()` | One gold CSV and one Step 2 prediction CSV | Single-pair score for one selected metric |
| `evaluate_step7_physical()` | Resolved-export manifest, gold manifest or canonical Step 2 gold, and explicit table correspondence | Physical-fragment report; strips declared appended columns |
| `score_table_grids()` | Prepared or raw gold and prediction rows | Shared per-fragment scores and sufficient statistics |
| `aggregate_grid_results()` | Per-fragment results from that scorer | Macro, micro, pooled counts, coverage, and population |
| `tabulus evaluate-table-batch` | Explicit-correspondence pair manifest | `summary.json` with both aggregations and per-fragment results |
| `tabulus aggregate-table-evaluations` | Completed, disjoint batch run directories with compatible settings | Recomputed macro/micro summaries and pooled counts |
| `scripts/evaluation/step2/evaluate_exact_match.py` | Discovers saved controlled, explicitly aligned production, or explicitly aligned physical Step 7 outputs in TabulusBench | Benchmark reports built on the same batch evaluator |

The first two interfaces prepare inputs for the same scoring core. The batch
CLI uses `evaluate_pairs()` to score explicit pairs and aggregate them;
the benchmark script discovers saved inputs and supplies those pairs. It does
not run reconstruction or resolved export.

`prepare_grid(rows)` caches the original rows and their RMS mapping preparation.
`score_table_grids(gold, prediction, metrics=...)` accepts rows or prepared grids
and returns scores, sufficient statistics and physical population counts.

```python
from tabulus.evaluation import prepare_grid, score_table_grids, aggregate_grid_results

gold = prepare_grid([["material", "value"], ["Al2O3", "300"]])
result = score_table_grids(
    gold, [["material", "value"], ["Al2O3", "300.0"]],
    metrics=("rms", "exact-mapping", "exact"),
)
summary = aggregate_grid_results([result])
```

`evaluate_table_reconstruction()` uses this core and retains its existing
metric names, score fields and defaults. Its additive `statistics` field
contains the selected comparison's credit, denominators and orientation.

Public preparation and execution interfaces also include `GridInput`,
`FragmentPair`, `load_grid()`, `prepare_step7_pairs()`,
`evaluate_pairs()` and `aggregate_evaluation_runs()`.

## Metric meanings

| Selection | Match credit | Denominators |
| --- | --- | --- |
| `rms` | Existing RMS key/value similarity after its key-based optimal assignment | Normalized mappings |
| `exact-mapping` | Equal normalized key and equal normalized string value | Normalized mappings |
| `exact` | Equal nonempty normalized value at the same physical coordinate | Nonempty physical cells |

RMS and Exact Mapping Match share CSV normalization, dictionary mapping
construction and duplicate-key handling. Later values overwrite earlier values
with the same normalized key. Blank values still contribute mappings. Their
mapping counts can differ from physical-cell counts.

RMS retains text and numeric thresholds, numeric comparison, assignment and
orientation behavior. Exact Mapping Match compares numeric strings literally.
Each mapping metric selects its own prediction orientation by highest F1.
Ties retain the historical preference for the transposed orientation.

Positional Exact Cell Match retains Unicode NFC and whitespace normalization,
case and punctuation sensitivity, original orientation, and whole-grid equality.
It counts headers as physical cells. Mapping metrics use row labels and column
headers as keys.

The optional evaluation dependency RapidFuzz accelerates the same exact
unit-cost Levenshtein distance. A Python fallback remains available. Backend
versions are recorded for checkpoint compatibility.

## Macro and micro

All public reports use the 0–100 score scale. Macro precision, recall and F1
are independent arithmetic means over the selected fragment population.
Missing, empty and unusable predictions contribute zero macro scores.

Micro pools sufficient statistics from the independently selected comparisons.

- RMS pools similarity credit and gold/prediction mapping counts.
- Exact Mapping Match pools exact mapping matches and mapping counts.
- Positional Exact Cell Match pools matching nonempty cells and cell counts.

For pooled credit `C`, gold denominator `G` and prediction denominator `P`,
micro precision is `C/P`, recall is `C/G`, and F1 is their harmonic mean.
F1 is computed from pooled counts. The implementation does not average or
size-weight fragment F1 to obtain micro F1.

Missing or unusable predictions retain `G`, with `P=0` and `C=0`.
They penalize micro recall without adding an artificial precision denominator.
Malformed CSV data is unusable. Invalid metadata declarations or correspondence
raise an error.

Mapping-level micro follows the existing mapping empty conventions. Both
denominators zero gives 100/100/100. With only the prediction denominator zero,
precision is 100, recall is zero and F1 is zero. With only the gold denominator
zero, precision is zero, recall is 100 and F1 is zero. Positional exact uses
zero for empty denominators. Thus all-missing macro F1 can be zero even when an
empty mapping population has vacuous micro F1 of 100. Coverage and pooled
counts must accompany these scores.

An empty prepared prediction has no non-whitespace physical cells. A valid
grid whose mapping representation is empty retains the mapping conventions.
Historical RMS statuses do not determine current prediction availability.

`pooled_counts` labels units as `mappings` or `nonempty_cells`. Report RMS
and Exact Mapping Match results as mapping-level micro scores.
Micro-averaging changes only the aggregation: larger fragments can contribute
more counts, so micro F1 may be higher or lower than macro F1. It does not
change the quality of any reconstruction or export.

## Explicit dataset-independent batches

A pair manifest describes the correspondence directly. CSV paths can be
absolute or relative to the manifest.

For a small illustration, suppose both gold CSVs contain `item,value` and
`A,1` as their two rows. The first prediction is identical; the second is
missing. A pair manifest makes that correspondence explicit:

```json
{
  "schema_version": 1,
  "pairs": [
    {
      "fragment_id": "example/fragment-a",
      "group": "example",
      "gold": {"csv": "gold/fragment-a.csv"},
      "prediction": {"csv": "predictions/fragment-a.csv"}
    },
    {
      "fragment_id": "example/fragment-b",
      "group": "example",
      "gold": {"csv": "gold/fragment-b.csv"},
      "prediction": null
    }
  ]
}
```

`GridInput` can declare `metadata_columns` and `header_row_index` for an
enriched side. Raw Step 2 gold has no metadata declaration. A missing prediction
can be represented by `null`. Archived failures can use
`prediction_status` with a value other than `ok`.

```bash
tabulus evaluate-table-batch --manifest pairs.json --out evaluation/run-a \
  --metrics rms exact-mapping exact --workers 1
tabulus evaluate-table-batch --manifest pairs.json --out evaluation/run-a \
  --metrics rms exact-mapping exact --workers 2 --resume
```

The first command produces all three metrics and both aggregations in one
`evaluation/run-a/summary.json`; `--resume` continues a compatible interrupted
run. A code-checked excerpt for the illustrative inputs above, selecting the
`exact` metric from `aggregate`, is:

```json
{
  "macro": {"exact": {"precision": 50.0, "recall": 50.0, "f1": 50.0}},
  "micro": {"exact": {"precision": 100.0, "recall": 50.0, "f1": 66.66666666666666}},
  "pooled_counts": {"exact": {"gold_count": 8, "predicted_count": 4, "matched_credit": 4.0, "unit": "nonempty_cells"}},
  "coverage": {"scored": 1, "missing": 1}
}
```

`summary.json` nests these fields under `aggregate` and also has `groups`.
The full report has the same fields for `rms` and `exact-mapping`: each has
gold mapping count 2, prediction mapping count 1, and matched credit 1 here.
Their mapping counts differ from the eight/four physical nonempty-cell counts.

Worker count can change on resume. Input hashes, correspondence, metric
settings, implementation hashes, Python version and relevant dependency
versions must remain compatible. Stale checkpoints raise an error.

Completed fragments are written incrementally using atomic replacements.
An OS lock prevents simultaneous writers to the same run directory and is
released after interruption. Only the parent process writes results.
Workers use one process-pool layer with numerical thread counts limited to one.
Gold preparation is cached per worker.

A run contains `run.json`, individual `fragments/*.json` checkpoints,
`results.json`, `summary.json` and `fragments.csv`.
Independent runs require independent output directories.
Final summaries are regenerated from sorted fragment identities, giving the
same aggregation and CSV order regardless of completion order or worker count.

The top-level aggregate counts evaluated pairs across groups. For benchmark
method comparisons use the individual groups and the unique full/selected
population statistics supplied by the benchmark adapter.

## Resolved-export manifest input

The generic `evaluate-table-batch` command above scores explicit pairs for
both Steps. For physical Step 7, a pair's resolved prediction declares its
appended `metadata_columns`; raw Step 2 gold declares none. The
{doc}`step7-quality` page documents the existing export-manifest input adapter,
which builds these pairs from `resolved_tables.json`, gold, and a reviewed
alignment. Its Step-specific command name identifies the input format, not a
different scorer or aggregation method.

## TabulusBench saved-output batches

The benchmark script validates the full 610-fragment population from
`metadata/tables.csv`. The generic metric and batch library contain no
benchmark population sizes or directory assumptions.

```bash
python scripts/evaluation/step2/evaluate_exact_match.py \
  --root "$HOME/tabulusbench" --out "$HOME/eval/step2-610" \
  --run-id run_01 --metrics rms exact-mapping exact --workers 4
```

The default metric remains `exact-mapping` for compatibility with older
commands. Use `--adapters` to select methods. `--no-native` disables automatic
MinerU-native inclusion. The native baseline reuses its archived correspondence
and saved output, with deterministic parsing of saved bodies for added
fragments when stable fragment key/page correspondence is unique. Its detected crops are
reported separately from canonical-crop adapters.

`--subset step7` selects the 61 reference-bearing P251–P253 fragments using
the physical Step 7 gold manifests. `--fragments subset.json` or a CSV with
`paper_id,table_key` selects an explicit subset.

The benchmark report includes each method's macro, micro, pooled counts and
coverage, plus full and selected population counts. `population_share`
reports fragment, nonempty-cell and gold-mapping fractions. Compare identical
fragment subsets when changing execution settings.

For production or resolved predictions, supply explicit correspondence.

```json
{
  "schema_version": 1,
  "tables": [
    {
      "adapter": "tesseract-tatr",
      "paper_id": "P251",
      "table_key": "page_026_table_018",
      "prediction_table_id": 20,
      "prediction_manifest": "/path/to/P251/resolved_tables.json"
    }
  ]
}
```

`prediction_manifest` is required for physical Step 7 so the metadata
declaration comes from the export. Production Step 2 can supply
`prediction_csv` explicitly or use its aligned prediction ID in the saved
production reconstruction directory. Paths in correspondence are relative
to its JSON file unless absolute. Omitting `adapter` applies a reviewed
mapping across the selected adapters. Each consumed prediction must map to
at most one gold fragment. Missing correspondence contributes a missing output.

```bash
python scripts/evaluation/step2/evaluate_exact_match.py \
  --root "$HOME/tabulusbench" --out "$HOME/eval/controlled-61" \
  --adapters tesseract-tatr --no-native --subset step7 \
  --metrics rms exact-mapping exact --workers 4

python scripts/evaluation/step2/evaluate_exact_match.py \
  --root "$HOME/tabulusbench" --out "$HOME/eval/production-61" \
  --adapters tesseract-tatr --no-native --subset step7 --setting production \
  --alignment /path/to/production-correspondence.json \
  --metrics rms exact-mapping exact --workers 4

python scripts/evaluation/step2/evaluate_exact_match.py \
  --root "$HOME/tabulusbench" --out "$HOME/eval/physical-step7-61" \
  --adapters tesseract-tatr --no-native --subset step7 --setting step7 \
  --alignment /path/to/resolved-correspondence.json \
  --metrics rms exact-mapping exact --workers 4
```

Append `--resume` to the identical command for a compatible interrupted run.
Benchmark summaries preserve the earlier `methods` reports and add group
aggregation, pooled counts, population statistics and resumable checkpoints.

### Historical archive coverage

Archived RMS reports can cover the original 605 fragments while current
evaluation covers all 610. Historical availability is separate from successful
scoring. Every selected fragment is evaluated using its actual gold and
prediction inputs, including P253 fragments absent from the older archive.
An absent historical entry never supplies a zero score or a missing-output flag.

Each method's `archive_verification` reports the archived population size,
selected population size, overlapping fragments, number checked, identifiers
`without_baseline`, score `mismatches` and `not_scored_with_baseline`.
RMS comparisons use an absolute tolerance of `1e-8`. Without RMS selected,
coverage is still reported and the numeric comparison is disabled. Archive
comparisons apply to controlled inputs and the separate native baseline.
Production and physical Step 7 require their own baselines for historical
verification. A discrepancy is reported alongside its archived value without
changing either score or source input.

## Slurm arrays and aggregation

Activate the evaluation environment and submit from the software checkout.
The examples request CPUs and memory without selecting a partition.

```bash
export BENCH_ROOT="$HOME/tabulusbench"
export EVAL_OUT="$HOME/eval/step2-array"
export EVAL_SUBSET=all
jobid=$(sbatch --parsable scripts/evaluation/step2/score_tables.slurm)
sbatch --dependency=afterok:"$jobid" scripts/evaluation/step2/aggregate_tables.slurm
```

The array runs one canonical-crop adapter per task with automatic native reuse
disabled. Set `EVAL_SUBSET=step7` for the matched 61-fragment population.
Each task resumes its own compatible checkpoints when present.

Completed disjoint runs can also be aggregated directly.

```bash
tabulus aggregate-table-evaluations \
  --runs "$EVAL_OUT"/task-* --out "$EVAL_OUT/aggregate"
```

Aggregation validates compatible settings and implementations and rejects
duplicate fragment identities. It recomputes pooled counts and macro/micro
summaries from fragment results. A native-only run can be submitted separately
using `--adapters mineru-native --no-native` and included explicitly among
the aggregation inputs. Source gold, predictions and production manifests
remain evaluation inputs.
