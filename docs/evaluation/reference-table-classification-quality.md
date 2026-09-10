# Reference-Table Classification Quality

Reference-table classification quality concerns Step 3: routing reconstructed
tables into or out of the reference-processing branch. It is separate from the
production classifier itself.

The implemented production classifier is deterministic. It reads the normalized
`parsed/` representation and `batch_summary.json` from one reconstruction
adapter directory, applies one regex/rule classifier to each reconstructed
structured table candidate, applies explicit continued-table inheritance when
available, and writes:

```text
<crop-root>/reconstructions/<adapter>/
  reference_table_classification.json
  selected_reference_tables.json
```

The public CLI for production classification is:

```bash
tabulus classify-reference-tables \
  --reconstruction /path/to/table-crops/<paper>/reconstructions/<adapter>
```

For crop-folder batches, the same production command can discover per-paper
reconstruction directories:

```bash
tabulus classify-reference-tables \
  --crops-folder /path/to/table-crops \
  --adapter <adapter>
```

## Current Implementation Status

Tabulus does not currently expose a library-native evaluator or `tabulus`
evaluation command for reference-table classification quality. The supported
library code implements the classifier and its persisted manifests, not a
classification scoring API.

## Scored Artifact

A classifier evaluator would score Step 3 table-level decisions, especially
`is_reference_table`, against curated labels. The scored artifact is the
classification manifest or selected-table manifest, not the prediction CSV
content itself and not later reference-matching or scholarly-resolution output.

## Metrics And Label Requirements

Recall is meaningful when the evaluation set contains only positive
reference-table examples: it measures how many known reference tables were
selected.

Precision, F1, specificity, and conventional accuracy require suitable negative
examples as well as positive examples. Without non-reference tables in the
label set, those metrics cannot be derived honestly.

Do not conflate reference-table classification with structural table
categorization, adapter difficulty subclasses, or table reconstruction quality.
Those questions use different labels and different denominators.
