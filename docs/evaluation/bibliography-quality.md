# Bibliography / Reference Extraction Quality

Bibliography/reference extraction quality evaluates whether extracted
bibliography entries match curated bibliography entries. It is separate from
table reconstruction, Stage 5 table-cell-to-bibliography matching, DOI
validation, and Stage 6 scholarly-identity resolution.

In the production pipeline, Stage 4 runs GROBID on the original PDF and writes
`references/bibliography.json`. That artifact is immutable extraction evidence
for downstream stages. It is not a scholarly-resolution artifact.

## Current Implementation Status

Tabulus currently implements bibliography extraction as production pipeline
code and exposes it through:

```bash
tabulus extract-bibliography \
  --pdf /path/to/paper.pdf \
  --out /path/to/artifact-root \
  --grobid-url http://localhost:8070
```

The public `src/tabulus.evaluation` package does not currently include a
library-native bibliography/reference extraction evaluator, and the CLI does
not expose a `tabulus` bibliography-evaluation command.

## Retained Research Utilities

The top-level `evaluation/scripts/` directory retains historical reference
extraction evaluators. They are not the public library API.

Those scripts include procedures for:

- fuzzy one-to-one comparison of predicted and gold GROBID-style reference
  strings using normalized text similarity;
- exact `(nr, ref)` comparison for older Kreuzberg-plus-regex extraction
  outputs;
- sliding-window similarity between curated references and raw OCR-extracted
  bibliography text across configured thresholds.

These retained utilities compute true positives, false positives, false
negatives, precision, recall, and F1 where their gold and prediction formats
support those counts. Some historical scripts name a found-reference ratio
`accuracy`; document that quantity as reference recovery rate unless a suitable
classification denominator exists.

## Metrics And Gold Requirements

A bibliography extraction evaluator needs a curated gold list of bibliography
entries and a prediction list from the extractor being scored. Matching may be
exact or similarity-based, and similarity thresholds must be reported with the
result.

Entry count agreement and reference recovery rate are useful diagnostics, but
they are not substitutes for precision, recall, or F1. Conventional accuracy
requires an explicit universe of positive and negative decisions; ordinary
bibliography extraction comparisons usually do not provide that denominator.

## Boundary

Bibliography/reference extraction evaluation scores extracted bibliography
entries. It does not score:

- whether a reconstructed table was selected for reference processing;
- whether a table cell was linked to the intended bibliography index;
- whether a DOI was validated;
- whether Stage 6 accepted a scholarly identity;
- resolution coverage across paper-level referenced bibliography entries.

Evaluation should read extraction artifacts and write separate metrics. It
should not modify `references/bibliography.json`, prediction CSV files,
`references/reference_matches.json`, or `references/reference_resolution.json`.
