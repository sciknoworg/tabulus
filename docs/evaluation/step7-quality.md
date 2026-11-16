# Step 7 Resolved-Table Quality

Step 7 has two public evaluation interfaces. The physical evaluator scores the
**original reconstructed grid carried through the default resolved CSV export**.
The optional structural/ORKG evaluator scores annotated projection properties.
The [resolved CSV contract](../data-contracts/resolved-csv.md) and
{doc}`../data-contracts/step7-structure` describe their distinct outputs.
Neither physical-grid metric measures the correctness of appended scholarly
metadata or citation-to-bibliography associations.

## Physical-Grid Inputs And Command

The library API is `tabulus.evaluation.evaluate_step7_physical(prediction_path,
gold_path, *, alignment_path=None, metric="both", text_threshold=0.5,
number_threshold=0.1)`. The CLI accepts one prediction manifest and either a
gold manifest or a benchmark paper root:

```bash
tabulus evaluate-step7-physical \
  --prediction /path/to/resolved_tables.json \
  --benchmark-paper /path/to/P251 \
  --alignment /path/to/alignment.json \
  --metric both \
  --out /path/to/scores.json
```

`--benchmark-paper` reads `step7_resolved_tables/gold.json` under that root.
Alternatively use `--gold /path/to/step7_resolved_tables/gold.json`; the two
options are mutually exclusive. `--alignment` is optional only when every
prediction entry already carries its stable gold `table_key`. `--metric` is
`both` (default), `exact`, or `rms`. `--text-threshold` and
`--number-threshold` configure RMS; `--out` is optional and otherwise the
report prints to standard output. Save reports outside the gold directory and
do not overwrite any input. RMS requires the optional evaluation dependencies
(`pip install 'tabulus[evaluation]'`).

Both manifests must be schema-version-1 JSON objects with a `tables` list.
The prediction `resolved_tables.json` declares `enrichment_columns`; each
table has an integer `table_id`, `resolved_csv`, and optional
`header_row_index`/stable `table_key`. The gold `gold.json` declares
`quality_columns`; each table has a unique stable `table_key`, `gold_csv`,
and optional `header_row_index`. These manifests point to the CSVs: the
evaluator reads the referenced grids, rather than treating the JSON alone as
table content. Gold and prediction may declare different metadata columns.

Prediction numeric table IDs are not assumed to equal gold fragment IDs. An
explicit, reviewed correspondence has this format:

```json
{
  "schema_version": 1,
  "tables": [
    {"table_id": 20, "table_key": "page_026_table_018"}
  ]
}
```

Every prediction entry must map to exactly one gold key. Duplicate mappings,
additional/unmapped prediction entries, and invalid keys are errors; an
unpredicted gold entry remains a scored missing fragment. The evaluator does
not infer correspondences from a paper name or numeric ID.

Relative `gold_csv` and `resolved_csv` paths resolve from their **own manifest
directories**; absolute paths resolve as written. A relocated manifest whose
CSV paths no longer point to the files must be corrected in a separate working
copy with verified paths before evaluation. `--alignment` maps identities, not
filesystem paths; it does not repair stale paths or alter the original gold.

## Physical-Grid Scoring

The evaluator removes each side's declared **trailing** metadata columns
before scoring. It validates a declared header row's metadata-column suffix
when `header_row_index` is present. It does not insert a header or treat row 0
as a universal header. The original physical grid, including any header rows,
citation text, and blank cells, remains the comparison unit. The default
physical export preserves the Step 2 rows and appends metadata arrays; it does
not create a physical row for every citation.

| Metric | Comparison | Orientation |
| --- | --- | --- |
| **Position-based Exact Cell Match** (`exact`) | Nonempty cells match only at the same row and column after Unicode NFC and whitespace collapse; case and punctuation remain significant. Precision/recall divide matches by nonempty predicted/gold cells. | Supplied physical coordinates only. |
| **Relative Mapping Similarity** (`rms`) | DePlot-style row-label/column-header mappings with textual similarity, numeric tolerance, and optimal assignment. | Scores original and transposed prediction, retaining the higher F1. |

Both return precision, recall, and F1 on `[0,100]`; RMS thresholds default to
`0.5` for text and `0.1` for numbers. RMS interprets its first input row as
headers during mapping conversion, independently of the manifest's optional
header-validation index. Exact Cell Match counts only nonempty matching cells,
while its separate `exact_grid` boolean compares the **whole normalized grid**,
including blank positions and row widths. `exact_grid_matches` counts exported
fragments with that boolean set when `exact` is requested.

An absent prediction entry or CSV has status `missing`; a readable CSV with no
rows has status `empty`; other readable exports have status `scored`. Missing
and empty exports contribute zero to the requested scores. For every gold
fragment, the report records `table_key`, prediction ID and paths, status,
metric scores, and `exact_cells` counts when requested. `coverage` reports
expected/exported/missing/empty counts. `macro` reports unweighted means of
fragment precision, recall, **and F1 separately**; macro F1 is not computed
from macro precision and recall. `inputs` records paths and SHA-256 hashes
(null for a missing CSV), alongside the score scale, normalization, thresholds,
and excluded-column lists.

These benchmark missing-output rules differ from some single-pair Step 2
empty-mapping conventions. The [Step 2 metric page](table-extraction-quality.md)
defines RMS, Exact Mapping Match (`--metric exact-mapping`), and Position-based
Exact Cell Match (`--metric exact`) separately. Exact Mapping Match equates
normalized row-label/column-header key/value mappings; Step 7 `exact` equates
physical cell coordinates after metadata removal. They are not interchangeable.
For saved Step 2 batch-scoring instructions, use that page rather than feeding
resolved CSVs into the Step 2 evaluator.

Current TabulusBench Step 7 physical gold covers P251–P253: 44, 13, and 4
reference-bearing fragments (61 total). The broader benchmark has 253 papers
and 610 controlled fragments. A Step 2/Step 7 comparison is valid only on
matched fragments with compatible gold grids, identical metric definitions,
and the same missing-output policy. Differences between separately published
scores alone cannot establish which Step caused an error.

## Scholarly Metadata Recovery In Citation Cells

Physical-grid scores deliberately remove scholarly metadata. A separate
**TabulusBench-side** analysis compares metadata values inside independently
aligned gold/production citation cells. Its retained
`scripts/step7/evaluate_step7_metadata.py` belongs to TabulusBench, not the
Tabulus package; no `tabulus evaluate-step7-metadata` command exists here.

For each aligned cell and metadata field (DOI, title, authors, year, venue),
the retained script reads named CSV columns containing JSON arrays, normalizes
values with Unicode NFKC, collapsed whitespace and case folding, then compares
value **multisets**. Repeated values count as repeated occurrences. Each
field's recovery denominator contains only asserted, nonempty gold values;
gold citation cells lacking a production correspondence contribute unrecovered
values. Values in unaligned production cells are counted separately rather
than assigned a gold identity. A table with an aligned citation cell is not
the same as a table merely exported to CSV.

A matching value multiset within a cell does **not** establish that each value
was attached to the correct bibliography entry. Entry-level identity accuracy
requires a validated bibliography-index correspondence. The expanded
P251–P253 gold contains 2,070 citation cells; the retained metadata-analysis
script and its report currently cover P251/P252 only (2,016 cells). Treat
their results as that narrower analysis, not as expanded-benchmark results.

## Optional Structural/ORKG Evaluation

The separate library API `tabulus.evaluation.step7.evaluate_step7()` and CLI
`tabulus evaluate-step7` evaluate the opt-in `--orkg` projection:

```bash
tabulus evaluate-step7 \
  --prediction /path/to/orkg/manifest.json \
  --benchmark-paper /path/to/P251 \
  --alignment /path/to/structural-crosswalk.json \
  --out /path/to/structural-scores.json
```

Use `--gold` instead of `--benchmark-paper` for an independently annotated
schema-version-1 development or held-out gold file. The optional structural
crosswalk maps table keys, source row/column coordinates, and bibliography
indices when namespaces differ; its format is **not** the physical evaluator's
`tables` list. This evaluator checks annotated routing, structural
correspondence, explicit/inherited citation links, scholarly identity fields
on aligned links, projection CSV integrity, coverage, and usage separately.
See {doc}`../data-contracts/step7-structure` for its input contract.

Benchmark-derived ORKG headers and available row provenance are **development
diagnostics**. Routing, header hierarchy, propagation, and held-out quality
require independent annotations. Automated tests verify the interface and
invariants, not live-model validation or benchmark accuracy. Optional
continuation merging is another distinct output: successful logical merges
retain the default physical CSVs and preserve physical-table provenance.
