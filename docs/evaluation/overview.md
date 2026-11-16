# Evaluation Overview

Evaluation reads gold and prediction artifacts without changing production
outputs. Each public command scores one pair (or, for Step 7, one paper-level
manifest pair) and optionally writes a separate JSON report with `--out`.
Corpus aggregation and experimental reports are separate from these library
interfaces.

## Public Evaluation Interfaces

| Step | Library evaluator | CLI | Prediction |
| --- | --- | --- | --- |
| [1: Table localization](table-localization-quality.md) | `evaluate_table_localization()` | `tabulus evaluate-table-localization` | `tables_index.json` |
| [2: Table reconstruction](table-extraction-quality.md) | `evaluate_table_reconstruction()` | `tabulus evaluate-table-reconstruction` | prediction CSV |
| [3: Reference-table classification](reference-table-classification-quality.md) | `evaluate_reference_table_classification()` | `tabulus evaluate-reference-table-classification` | `reference_table_classification.json` |
| [4: Bibliography extraction](bibliography-quality.md) | `evaluate_bibliography()` | `tabulus evaluate-bibliography` | `references/bibliography.json` |
| [5: Reference matching](reference-matching-quality.md) | `evaluate_reference_matching()` | `tabulus evaluate-reference-matching` | `references/reference_matches.json` |
| [6: Scholarly identity resolution](reference-resolution-quality.md) | `evaluate_reference_resolution()` | `tabulus evaluate-reference-resolution` | `references/reference_resolution.json` |
| [7: Original physical grids](step7-quality.md) | `evaluate_step7_physical()` | `tabulus evaluate-step7-physical` | `resolved_tables.json` and physical CSVs |
| [7: Optional structural/ORKG projection](step7-quality.md) | `tabulus.evaluation.step7.evaluate_step7()` | `tabulus evaluate-step7` | `orkg/manifest.json` and projection CSVs |

The Step 7 interfaces answer different questions. Physical-grid evaluation
scores original table content after export; it excludes appended metadata.
Structural/ORKG evaluation scores annotated projection properties, citation
links, identity fields, and coverage. Neither replaces the Step 2 score of a
raw reconstruction.

## Gold, Predictions, And Evaluation Questions

Gold is a Step-specific truth source, kept separately from prediction/run
directories. A **controlled run** uses available gold upstream inputs as
needed to isolate a Step. A **full-corpus run** executes the production
workflow over the broader collection. Gold type and run category are
independent.

These runs support distinct evaluation questions:

- **Controlled component quality:** compare one Step with its own gold while
  limiting upstream error propagation.
- **Production coverage:** count produced, missing, empty, rejected, or
  unmatched outputs over the applicable corpus. Coverage and Step 6
  resolution yield are operational outcomes, not independently verified
  correctness.
- **Gold-aligned end-to-end quality:** align production outputs with the
  appropriate gold and score the resulting content. This requires explicit
  correspondence and compatible denominators; it is not another run category.

For example, Step 6 source identity gold and a production Step 4 bibliography
use different index namespaces unless a validated one-to-one alignment or
crosswalk connects them. A stored `step4_index` alone does not prove that
correspondence. Missing source DOI is unknown, not proof that a DOI does not
exist. See {doc}`reference-resolution-quality`.

## Canonical TabulusBench Gold Layout

Gold lives below each paper root; prediction artifacts live under
`tabulus_runs/` or another user-selected work directory:

```text
<paper_root>/
├── step1_table_localization/gold.json
├── step2_table_reconstruction/tables/<table_id>/gold.csv
├── step3_reference_table_classification/gold.json
├── step4_bibliography/gold.json                 # where annotated
├── step5_reference_matching/gold.json           # deep subset
├── step6_reference_resolution/gold.json         # deep subset
└── step7_resolved_tables/
    ├── gold.json                                 # deep subset
    └── tables/<table_id>/gold.csv
```

Step 1 gold also has `crops/`. The optional ORKG development annotations are
separate under `step7_orkg_gold/`. Tabulus-Gold uses human-readable paper names;
TabulusBench uses IDs such as `P251`. Evaluators do not infer identity from a
directory basename.

## Gold Coverage And Denominators

The current TabulusBench catalog contains **253 papers and 610 controlled
physical fragments**. Coverage varies by Step:

| Step | Available gold | Interpretation |
| --- | --- | --- |
| 1 | 253 papers; 610 physical fragments | `pseudo_gold` localization |
| 2 | 253 papers; 610 curated `gold.csv` tables | raw reconstruction |
| 3 | 253 papers; 610 binary labels (578 positive, 32 negative) | `pseudo_gold` routing |
| 4 | 53 papers; 8,004 numbered references | authoritative bibliography gold |
| 5 | P251–P253; 61 reference-bearing fragments, 2,070 citation cells, 6,817 link occurrences | `derived_gold` links |
| 6 | P251–P253; 4,075 source-derived identity records (2,280 / 996 / 799) | independently sourced scholarly metadata; evaluate only after namespace alignment |
| 7 | P251–P253; 61 reference-bearing physical output fragments (44 / 13 / 4) | `derived_gold` resolved tables |

The 578/610 Step 3 ratio describes the deliberately reference-focused selected
fragments, not all tables in the source PDFs or a scientific domain. The Step 5
and Step 7 deep subset is smaller than the full benchmark. Gold citation-cell
counts do not imply that every cell was aligned to a production output.

Older 252-paper/605-fragment results retain their historical population and
must be labeled as such. The [Step 2 batch scorer](table-extraction-quality.md)
uses the expanded 610-fragment set. Some retained TabulusBench Step 7 metadata
reports and scripts still cover only P251/P252 (57 fragments and 2,016 citation
cells); they are not expanded P251–P253 results.

## Why There Is No Single Pipeline Accuracy

Step 1 measures fragment recovery, Step 2 reconstructs physical grids, Step 3
routes tables, Step 4 recovers bibliography entries, Step 5 links citation
cells, Step 6 checks scholarly identities, and Step 7 scores final physical
grids or optional structural projections. Each has its own gold, metric, and
denominator. A resolved CSV is not a Step 2 prediction CSV.

Step 2 Exact Mapping Match compares normalized row-label/column-header
mappings; its Position-based Exact Cell Match compares the same physical cell
coordinates. Step 7 physical `exact` scores coordinates after stripping the
declared appended columns, while its `rms` scores mappings. See the
{doc}`table-extraction-quality` metric definitions and {doc}`step7-quality`
for the export-specific policy. Comparing published Step 2 and Step 7 scores
requires matched fragments, compatible gold grids, the same metric, and the
same missing-output policy. Different scores alone do not identify where an
error occurred; paired error analysis remains separate work.
