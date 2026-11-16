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

The gold JSON contains a `records` list. The evaluator joins a record's
`gold_index` to the prediction `resolution.reference_index`; it falls back to
the historical `step4_index` field only when `gold_index` is absent. The
prediction JSON contains an `entries` list. This API performs an **index join**,
not a bibliography-alignment check. Its comparison is valid only when the
prediction indices have already been independently verified to refer to the
same gold bibliography positions. A stored `step4_index` is provenance, not
evidence that a production Step 4 entry still represents the same work.

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

Canonical Step 6 source-derived gold records currently cover P251–P253:

- P251: 2,280 source-derived scholarly identities
- P252: 996 source-derived scholarly identities
- P253: 799 source-derived scholarly identities
- total: 4,075 records

P251/P252's 3,276 records form the earlier controlled one-to-one eligible
identity subset. The expanded gold also includes P253. Source-annotated
identities are not automatically the denominator for every production run:
first establish which prediction bibliography entries correspond one-to-one to
the gold `gold_index` namespace, or apply a separately validated crosswalk.
Split, merged, and unmatched Step 4 entries must remain outside that clean
component comparison and in the upstream alignment accounting. The public
pair-level evaluator does not construct or validate this crosswalk itself.

The identity population is independently source-derived, not simply all Step 5
links. Uncovered links are not automatically resolver errors. A controlled
Step 6 run with verified gold-index inputs can be scored against these records;
production predictions require the same correspondence check first. Resolution
yield and `validated_with_doi` are output rates, not identity accuracy labels.

## Boundary

Step 6 evaluation does not evaluate Step 5 citation-cell matching or the
Step 7 physical-grid export. See {doc}`step7-quality` for final export and
optional structural evaluation. Step 7 consumes already-resolved identities;
it does not re-resolve them.
