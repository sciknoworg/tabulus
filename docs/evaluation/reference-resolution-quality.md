# Step 6 Scholarly Identity Resolution Quality

Step 6 evaluation scores paper-level scholarly identity resolution against
independent source-derived identity gold. Step 6 resolves unique cited
bibliography entries at paper level after Step 5 has linked table citation
occurrences to bibliography positions.

The production artifact is {doc}`../data-contracts/reference-resolution-json`,
written as `references/reference_resolution.json`.

## Public Interface

The library API is `tabulus.evaluation.evaluate_reference_resolution()`. The CLI
is:

```bash
tabulus evaluate-reference-resolution \
  --gold /path/to/paper/step6_reference_resolution/gold.json \
  --prediction /path/to/artifact-root/references/reference_resolution.json \
  --out /path/to/evaluation/step6-reference-resolution.json
```

`--out` is optional. Evaluation reads the gold and prediction JSON files and
does not modify the Step 6 resolution artifact.

## Gold And Prediction Inputs

Canonical gold lives at:

```text
<paper_root>/step6_reference_resolution/gold.json
```

The gold JSON contains a `records` list. Controlled predictions are aligned by
`gold_index` when present, with `step4_index` retained as provenance and used as
a legacy fallback. The prediction JSON contains an `entries` list; each entry
must include a `resolution` object with an integer `reference_index`.

Prediction statuses counted by the evaluator are:

- `validated_with_doi`
- `validated_without_doi`
- `rejected`

`validated_with_doi` and `validated_without_doi` count as resolved targets for
resolution-yield reporting. `rejected` remains a scientific outcome, not an
operational failure.

## Metrics

The serialized result includes:

- `gold_targets`
- `matched_predictions`
- `missing_predictions`
- `resolved_targets`
- `rejected_targets`
- `resolution_yield`
- `doi_evaluable`
- `doi_predictions`
- `doi_exact_matches`
- `doi_precision`, `doi_recall`, `doi_f1`
- `title_evaluable`, `title_correct`, `title_accuracy`
- `authors_evaluable`, `authors_correct`, `authors_accuracy`
- `year_evaluable`, `year_correct`, `year_accuracy`
- `venue_evaluable`, `venue_correct`, `venue_accuracy`
- `status_counts`

Missing DOI in source metadata is unknown, not a negative DOI label. A prediction
is not penalized merely because source gold has no DOI field. DOI metrics are
computed only over gold records with an available DOI, while DOI precision uses
the count of available DOI predictions.

## Comparison Semantics

DOIs are normalized by removing DOI URL/prefix forms and trailing punctuation
before exact comparison. Titles are normalized to alphanumeric tokens and accept
exact equality or a SequenceMatcher ratio of at least `0.95`. Authors are
compared as ordered normalized surnames; string author lists split on `and` and
list-valued author fields are also supported. Years compare by exact string
value after trimming. Venue comparison uses normalized tokens, accepts common
prefix compatibility, and requires at least 80% token compatibility over the
shorter token list.

These metrics assess scholarly identity agreement. They are distinct from Step
6 operational diagnostics such as LLM usage, retry rate, provider failures, or
remote-service latency.

## TabulusBench Scope

Canonical Step 6 gold currently covers the P251/P252 deep subset:

- P251: 2,280 source-derived scholarly identities
- P252: 996 source-derived scholarly identities
- total: 3,276 source-derived scholarly identities

The Step 6 identity population is independently source-derived. It is not simply
all linked bibliography positions, and uncovered links should not automatically
be interpreted as resolver errors.

A controlled run of Step 6 can be evaluated against this gold. Current
regression-checked controlled-run reference points include:

- P251 resolution yield: 84.21%
- P252 resolution yield: 76.41%
- P251 DOI precision/recall/F1: 99.43% / 91.11% / 95.09%

These values identify particular controlled-run results and should not be
reported as universal end-to-end pipeline accuracy. Step 6 evaluation uses
source-derived identity gold; validation yield alone is not an accuracy label.

## Boundary

Step 6 evaluation does not evaluate Step 5 citation-cell matching or Step 7
resolved CSV export. Step 7 deterministically consumes the already-resolved Step
6 artifact and does not re-resolve scholarly identities.
