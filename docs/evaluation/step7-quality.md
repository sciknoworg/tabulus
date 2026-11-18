# Step 7 Resolved-Table Quality

Step 7 outputs support two evaluation questions. **Physical-grid evaluation**
checks whether the original reconstructed table content survived the default
resolved CSV export, using the same scorer as Step 2. **Structural/ORKG
evaluation** checks an optional projection. The physical score excludes
appended scholarly metadata and citation associations.
See the {doc}`../data-contracts/resolved-csv` and
{doc}`../data-contracts/step7-structure` contracts for the two output formats.

## Evaluate Physical Tables

### Batch scoring: macro and micro

Use the same batch command as Step 2 when gold and resolved CSVs have explicit
fragment correspondence:

```bash
tabulus evaluate-table-batch \
  --manifest /path/to/physical-pairs.json \
  --metrics rms exact-mapping exact \
  --out /path/to/physical-grid-evaluation
```

The pair manifest points to curated Step 2 `gold.csv` files and resolved CSVs.
On each resolved prediction, declare the appended `metadata_columns` from
`resolved_tables.json` so the shared scorer removes them before comparison.
The command writes per-fragment results and a `summary.json` containing both
macro and micro scores. See {doc}`physical-grid-evaluation` for the manifest
format and aggregation rules.

### Evaluate an export manifest directly

The existing `evaluate-step7-physical` command prepares those same fragment
pairs from a `resolved_tables.json` manifest and reviewed alignment. Its name
identifies this Step 7 input adapter; it does not implement separate metrics or
separate macro/micro averaging.

```bash
tabulus evaluate-step7-physical \
  --prediction /path/to/resolved_tables.json \
  --step2-gold /path/to/canonical-step2-gold \
  --alignment /path/to/alignment.json \
  --metrics rms exact-mapping exact \
  --out /path/to/scores.json
```

The library API is `tabulus.evaluation.evaluate_step7_physical()`.
Both Step 2 and this physical Step 7 interface support `rms`,
`exact-mapping`, and `exact`; `--metric both` remains the compatible default
for RMS plus positional exact. The export-manifest command also supports
parallel workers and resumable checkpoints.

| Option | Meaning |
| --- | --- |
| `--prediction` | Physical Step 7 `resolved_tables.json` manifest. |
| `--benchmark-paper`, `--gold`, or `--step2-gold` | Paper root containing `step7_resolved_tables/gold.json`, that gold manifest directly, or the canonical Step 2 directory with `tables_index.json` and `tables/`. Supply exactly one. |
| `--alignment` | Reviewed mapping from prediction table IDs to stable gold fragment keys. Omit only if every prediction entry already has its gold `table_key`. |
| `--metric` or `--metrics` | `both` (default) retains positional exact plus RMS. Use `--metrics rms exact-mapping exact` for all three. Optional `--text-threshold` and `--number-threshold` affect RMS only. |
| `--workers`, `--checkpoint-dir`, `--resume` | CPU parallelism and compatible incremental checkpoint reuse. |
| `--out` | Optional report path outside the gold directory and distinct from every input; without it, JSON prints to standard output. |

RMS needs the optional evaluation dependencies (`pip install
'tabulus[evaluation]'`).

### Manifests and alignment

Both manifests use schema version 1 and point to CSVs. The prediction manifest
declares `enrichment_columns` and, per table, an integer `table_id` and
`resolved_csv`. Gold declares `quality_columns` and, per table, a stable
`table_key` and `gold_csv`. Either side may include `header_row_index`; the
prediction may carry its own stable `table_key`. Each side may declare a
different set of appended columns.

When the prediction lacks gold keys, supply an alignment file:

```json
{
  "schema_version": 1,
  "tables": [{"table_id": 20, "table_key": "page_026_table_018"}]
}
```

Each prediction must map to one gold fragment. Duplicate mappings, extra or
unmapped predictions, and invalid keys are errors. Gold fragments without a
prediction remain in the report as missing. Numeric IDs and paper-directory
names do not establish correspondence.

Each relative CSV path resolves from **its own manifest directory**; absolute
paths resolve as written. If a copied manifest still points to its former
location, correct verified CSV paths in a separate working copy before scoring.
`--alignment` does not repair paths or modify benchmark gold.

### Scores and report

The evaluator first removes each side's declared trailing metadata columns.
Raw canonical Step 2 gold has no metadata removed.
It checks their names on a declared header row, if present. It does not insert
a header or assume row 0 is always one. The scored grid retains physical rows,
headers where present, citation text, and blank cells. Default Step 7 export
appends metadata arrays to the Step 2 grid; it does not add one row per citation.

- **`exact` — Position-based Exact Cell Match:** nonempty values match only at
  the same row and column after Unicode NFC and whitespace collapse. Case and
  punctuation remain significant. Precision and recall divide matches by
  nonempty prediction and gold cells, respectively. There is no transpose.
  Separately, `exact_grid` tests whole-grid equality, including blank positions
  and row widths.
- **`rms` — Relative Mapping Similarity:** DePlot-style row-label/column-header
  mappings use text similarity, numeric tolerance, and optimal assignment.
  The higher-F1 result from original or transposed prediction is retained.
  RMS treats its first input row as headers during mapping conversion,
  independently of `header_row_index` used for metadata validation.
- **`exact-mapping` — Exact Mapping Match:** the same normalized mapping
  representation requires equal keys and string values, with its own
  highest-F1 orientation selection.

All three metrics report precision, recall, and F1 on `[0,100]`. RMS defaults to
text threshold `0.5` and number threshold `0.1`. An absent prediction entry or
CSV is `missing`; a readable CSV with no non-whitespace cells is `empty`;
malformed CSV is `unusable`; other readable exports are `scored`.
Missing, empty and unusable exports score zero for macro aggregation.

The JSON report includes per-fragment paths, status, scores, and `exact_cells`
counts when `exact` is requested. It also records coverage
(`expected`/`exported`/`missing`/`empty`), `exact_grid_matches` when applicable,
excluded columns, normalization, thresholds, and input paths with SHA-256
hashes (null for a missing CSV). `exact_grid_matches` counts exported
fragments whose whole grids are equal when `exact` is requested.
The report also includes per-metric macro and micro scores and pooled counts.

**Macro** averages each physical fragment's precision, recall, and F1
separately. **Micro** pools match credit and gold/prediction counts first, then
calculates precision, recall, and F1 from the pooled totals. Missing exports
retain gold counts in micro recall. RMS and `exact-mapping` pool normalized
mappings; `exact` pools nonempty physical cells. The
{doc}`physical-grid-evaluation` guide gives the full denominator, missing-output,
and empty-population rules.

The {doc}`table-extraction-quality` page defines the three Step 2 metrics and
its saved-prediction batch scorer. Step 2 **Exact Mapping Match**
(`--metric exact-mapping`) compares normalized row-label/column-header mappings;
Position-based Exact Cell Match (`--metric exact`) compares physical
coordinates. Step 7 `exact` uses the latter after removing metadata. Single-pair
Step 2 empty-mapping conventions differ from Step 7's missing/empty-zero
policy; the Step 2 batch scorer separately assigns zero to missing or unusable
predictions. Do not pass enriched CSVs to the Step 2 evaluator.

TabulusBench Step 7 physical gold covers P251–P253: **44, 13, and 4**
reference-bearing fragments (61 total), within a benchmark of 253 papers and
610 controlled fragments. Comparing separate Step 2 and Step 7 scores requires
matched fragments, compatible gold grids, the same metric, and the same
missing-output policy. The full paired error analysis is still pending.

## Scholarly Metadata Recovery

Metadata recovery is a different question from physical-grid preservation. A
retained **TabulusBench-side** script,
`scripts/step7/evaluate_step7_metadata.py`, analyzes independently aligned
gold/production citation cells. It is not a Tabulus package API or CLI.

For DOI, title, authors, year, and venue, the script reads named CSV columns
containing JSON arrays. It normalizes values with Unicode NFKC, collapsed
whitespace, and case folding, then compares **multisets** within each aligned
cell. Repeated values remain repeated. Each field's denominator includes only
asserted, nonempty gold values; missing gold citation cells contribute unrecovered
values. Values in unaligned production cells are reported separately. Thus
“tables with aligned citation cells” differs from “exported tables.”

Equal value multisets do not prove that each value belongs to the correct
bibliography entry; that requires a validated index correspondence. Expanded
P251–P253 gold has 2,070 citation cells. The retained script and report cover
P251/P252 only (2,016 cells), so their results do not describe the expanded
population.

## Optional Structural/ORKG Evaluation

The separate `tabulus evaluate-step7` command and
`tabulus.evaluation.step7.evaluate_step7()` API score the optional `--orkg`
projection:

```bash
tabulus evaluate-step7 \
  --prediction /path/to/orkg/manifest.json \
  --benchmark-paper /path/to/P251 \
  --alignment /path/to/structural-crosswalk.json \
  --out /path/to/structural-scores.json
```

Use `--gold` instead of `--benchmark-paper` for independent schema-version-1
development or held-out annotations. The optional crosswalk can map table
keys, row/column coordinates, and bibliography indices; its format differs
from the physical evaluator's alignment list. The evaluator reports annotated
structure, citation links, identity fields on aligned links, projection CSV
integrity, coverage, and usage separately. See
{doc}`../data-contracts/step7-structure` for details.

Benchmark-derived ORKG headers and available row provenance are development
diagnostics. Independent annotations are needed for routing, header hierarchy,
propagation, and held-out quality. Automated tests establish interface
behavior, not live-model validation or benchmark accuracy. Optional
continuation merging is a separate Step 7 output that retains physical CSVs
and their provenance.
