# Step 3 Reference-Table Classification Quality

Step 3 evaluation scores binary `is_reference_table` decisions for reconstructed
physical tables. It is separate from Step 2 reconstruction quality: a table can
be reconstructed well but routed incorrectly, or reconstructed poorly but routed
correctly.

The production Step 3 artifact is `reference_table_classification.json`, written
beside the Step 2 reconstruction directory. `selected_reference_tables.json` is
the downstream pointer manifest for Step 5, but the pair-level classifier
evaluator scores the complete classification manifest.

## Public Interface

The library API is `tabulus.evaluation.evaluate_reference_table_classification()`.
The CLI is:

```bash
tabulus evaluate-reference-table-classification \
  --gold /path/to/paper/step3_reference_table_classification/gold.json \
  --prediction /path/to/reconstruction/reference_table_classification.json \
  --out /path/to/evaluation/step3-reference-table-classification.json
```

`--out` is optional. Evaluation reads the gold and prediction manifests and does
not modify Step 2 or Step 3 artifacts.

## Alignment And Labels

The evaluator expects each input to be a JSON object with a `tables` list.
Gold tables are aligned to predictions by `table_id` when possible. Prediction
keys are collected from `table_id`, `source_prediction`, and `source_parsed`.
If key alignment is incomplete but both lists have the same length, the evaluator
uses manifest order for controlled manifests that preserve table order.

Each aligned pair must provide boolean `is_reference_table` labels.

## Metrics

The result includes:

- `gold_tables`
- `positive_gold`
- `negative_gold`
- `true_positives`
- `false_positives`
- `true_negatives`
- `false_negatives`
- `precision`
- `recall`
- `f1`
- `specificity`
- `accuracy`
- `balanced_accuracy`

Specificity is `null` when the evaluated scope contains no negative examples.
Balanced accuracy is `null` unless both positive and negative classes are
present. Corpus-level balanced accuracy should be computed from the aggregated
confusion matrix across a complete labeled population, not by pretending an
absent class has zero recall.

## TabulusBench Scope

TabulusBench provides Step 3 labels for all 252 papers and 605 physical table
fragments:

- 574 reference-bearing tables
- 31 non-reference tables
- 0 unresolved labels

The gold type is `pseudo_gold`. The ratio `574/605 = 94.88%` is prevalence
within the deliberately curated, reference-focused TabulusBench table set. It
must not be interpreted as the prevalence of reference-bearing tables among all
tables in the 252 papers or within scientific domains.

The canonical gold path is:

```text
<paper_root>/step3_reference_table_classification/gold.json
```

Step 3 evaluation does not score the content of prediction CSVs, Step 5 citation
links, or Step 6 scholarly identities.
