# Evaluation Overview

Evaluation is operationally separate from the production pipeline.
Production steps write stable artifacts; evaluation reads those artifacts and,
when requested, writes separate metrics. Evaluation should not mutate
production artifacts such as prediction CSV files, `references/bibliography.json`,
`references/reference_matches.json`, or `references/reference_resolution.json`.

Tabulus documentation separates evaluation into three principal levels:

```text
reference-table classification labels
        ^
        |
classification evaluation
        |
selected/reference-table decisions

manual table CSV
        ^
        |
Relative Mapping Similarity (RMS)
        |
prediction CSV

curated bibliography entries
        ^
        |
reference-extraction comparison
        |
references/bibliography.json or retained extraction output
```

## Current Library-Native Evaluation

The current public `src/tabulus.evaluation` package implements native table
reconstruction evaluation only. The supported metric is Relative Mapping
Similarity (RMS), adapted from the DePlot table-datapoint metric. It is exposed
through the `tabulus evaluate-table-reconstruction` command and the
`evaluate_table_reconstruction()` API.

Reference-table classification and bibliography/reference extraction have
production pipeline steps, but they do not currently have library-native
public evaluators or `tabulus` evaluation commands.

## Retained Research Evaluation Utilities

The top-level `evaluation/` directory contains retained research scripts,
legacy metric harnesses, plots, and generated comparison material. Those files
are useful for understanding earlier experiments, but they are not the public
Tabulus library interface unless code under `src/tabulus` imports them or the
CLI exposes them.

Some retained scripts write result JSON into dataset folders. Treat them as
read-only with respect to production artifacts and inspect their arguments and
outputs before running them on any benchmark tree.

## Boundaries

Use the metric that matches the artifact being scored:

- {doc}`reference-table-classification-quality` scores whether Step 3 routed
  reconstructed tables into the reference-table branch.
- {doc}`table-extraction-quality` scores raw table reconstruction prediction
  CSVs against manually curated table CSVs.
- {doc}`bibliography-quality` scores bibliography/reference extraction against
  curated bibliography entries.
- {doc}`reference-matching-quality` describes Step 5 link diagnostics and the
  Step 6 resolution denominator, but neither Step 5 matching nor Step 6
  scholarly resolution currently has a native public evaluator.

Do not aggregate these levels into one pipeline accuracy number. Coverage,
consistency, agreement, recovery rate, precision, recall, F1, and resolution
coverage answer different questions and have different denominators.

Resolved CSV files are downstream Step 7 outputs. They are not the artifact
used to measure raw table reconstruction quality, because they append
reference-resolution fields to Step 2 prediction rows.
