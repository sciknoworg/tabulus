# Evaluation Overview

Evaluation is separate from production execution. Production commands write
pipeline artifacts such as `tables_index.json`, prediction CSVs,
`references/bibliography.json`, `references/reference_matches.json`, and
`references/reference_resolution.json`. Evaluation commands read gold and
prediction artifacts and may write a separate JSON result with `--out`; they do
not modify the artifacts being scored.

## Step Map

| Pipeline Step | Evaluator | CLI | Scored prediction |
| --- | --- | --- | --- |
| Step 1 -- Table localization | `evaluate_table_localization()` | `tabulus evaluate-table-localization` | `tables_index.json` |
| Step 2 -- Table reconstruction | `evaluate_table_reconstruction()` | `tabulus evaluate-table-reconstruction` | prediction CSV |
| Step 3 -- Reference-table classification | `evaluate_reference_table_classification()` | `tabulus evaluate-reference-table-classification` | `reference_table_classification.json` |
| Step 4 -- Bibliography extraction | `evaluate_bibliography()` | `tabulus evaluate-bibliography` | `references/bibliography.json` |
| Step 5 -- Reference matching | `evaluate_reference_matching()` | `tabulus evaluate-reference-matching` | `references/reference_matches.json` |
| Step 6 -- Scholarly identity resolution | `evaluate_reference_resolution()` | `tabulus evaluate-reference-resolution` | `references/reference_resolution.json` |

The current public pair-level evaluation endpoint set covers Steps 1--6. Step 7
is deterministic resolved-table export and has canonical derived gold for the
P251/P252 deep subset, but there is not currently a public Step 7 evaluation
CLI endpoint in this six-command interface.

## Gold, Prediction, And Run Categories

Gold and prediction artifacts are different things. Gold is the independent
truth source for one evaluation question. Prediction artifacts are outputs from
Tabulus production commands or controlled-run inputs produced for component
isolation.

There are two run categories in the documentation:

`controlled run`
: Apply a Step where the requisite gold-standard data for that Step exists,
  using gold upstream information as needed to isolate that Step from upstream
  errors.

`full-corpus run`
: Production or end-to-end execution over the broader corpus.

Gold type and run category are independent. For example, a controlled run may
use `pseudo_gold`, `derived_gold`, authoritative bibliography gold, or
source-derived scholarly-identity gold depending on the Step. Gold should remain
conceptually independent of `tabulus_runs/` directories.

## Canonical TabulusBench Gold Layout

Canonical TabulusBench gold lives below each paper root using Step-based names:

```text
<paper_root>/
├── step1_table_localization/
│   ├── gold.json
│   └── crops/...
├── step2_table_reconstruction/
│   └── tables/<table_id>/gold.csv
├── step3_reference_table_classification/
│   └── gold.json
├── step4_bibliography/
│   └── gold.json
├── step5_reference_matching/
│   └── gold.json
├── step6_reference_resolution/
│   └── gold.json
└── step7_resolved_tables/
    ├── gold.json
    └── tables/<table_id>/gold.csv
```

Prediction and run artifacts live separately under `tabulus_runs/` or another
user-selected work directory. Older non-Step-based gold paths describe
historical material and should not be used for new public examples.

Tabulus-Gold and TabulusBench deliberately use different paper names. Tabulus-
Gold may use human-readable names such as `Miikkulainen_2013` or
`Puurunen_2005`; TabulusBench uses identifiers such as `P251` and `P252`. The
pair-level evaluators do not infer evaluation semantics from a paper-directory
basename.

## TabulusBench Coverage

The benchmark has different gold coverage by Step:

| Step | Gold coverage | Gold type / interpretation |
| --- | --- | --- |
| Step 1 | 252 papers; 605 physical table fragments/crops | `pseudo_gold` physical-fragment recovery gold |
| Step 2 | 252 papers; 605 curated canonical `gold.csv` tables | curated reconstruction gold |
| Step 3 | 252 papers; 605 binary labels: 574 reference-bearing, 31 non-reference, 0 unresolved | `pseudo_gold` reference-table labels |
| Step 4 | 52 papers; 6,947 numbered gold bibliography references | authoritative bibliography gold |
| Step 5 | P251 and P252 only; 57 reference-bearing physical fragments; 2,016 citation cells; 5,963 bibliography-link occurrences | `derived_gold` citation-link gold |
| Step 6 | P251 and P252 only; 3,276 source-derived scholarly identities | source-derived authoritative identity gold |
| Step 7 | P251 and P252 only; 57 resolved output fragments | `derived_gold` resolved-table gold |

The Step 3 ratio `574/605 = 94.88%` is prevalence within the deliberately
curated, reference-focused TabulusBench table set. It is not the prevalence of
reference-bearing tables among all tables in the 252 source papers or in any
scientific domain.

The deep P251/P252 subset progresses through different denominators:

```text
57 reference-bearing fragments
→ 2,016 citation cells
→ 5,963 citation-link occurrences
→ 3,341 unique bibliography entries linked
→ 3,276 source-derived scholarly identities
→ 57 resolved output fragments
```

Do not treat uncovered Step 6 links as errors by default. The Step 6 identity
population is independently source-derived.

## Why There Is No Single Pipeline Accuracy

Each Step answers a different question:

- Step 1 measures physical-fragment recovery/alignment, not bounding-box IoU.
- Step 2 measures raw table reconstruction with Relative Mapping Similarity
  (RMS), before reference enrichment.
- Step 3 measures binary routing of reconstructed tables into the
  reference-processing branch.
- Step 4 measures bibliography-entry extraction and alignment from the original
  PDF.
- Step 5 measures citation-cell links to bibliography positions.
- Step 6 measures source-derived scholarly identity agreement.
- Step 7 is deterministic export from existing Step 5 and Step 6 artifacts.

Resolved CSV files should not be used as Step 2 reconstruction predictions,
because Step 7 intentionally appends and filters resolution fields. Coverage,
precision, recall, F1, resolution yield, strict-entry matching, and RMS F1 have
different denominators and should be reported with their Step context.
