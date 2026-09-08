# Evaluation

This directory contains retained research evaluation utilities, plots, and
legacy benchmark material from earlier Tabulus experiments. It is not the
public Tabulus evaluation package.

The supported library evaluation surface lives under `src/tabulus/evaluation`.
At present, that public package exposes native table reconstruction evaluation
with Relative Mapping Similarity (RMS), and the `tabulus` CLI exposes it through
`tabulus evaluate-table-reconstruction`.

Files in this directory should be treated as historical or research utilities
unless current code under `src/tabulus` imports them or a documented `tabulus`
command calls them.

## Retained Utility Areas

The retained scripts cover several older evaluation procedures:

- table CSV comparison using the DePlot table-datapoint metric;
- batch traversal of historical prediction folders;
- fuzzy one-to-one comparison between predicted and gold bibliography entries;
- exact `(nr, ref)` comparison for older reference-extraction outputs;
- raw bibliography-text reference recovery using sliding-window similarity;
- table-reference coverage checks against historical table-reference lists.

These procedures use different artifacts and denominators. Do not combine them
into one pipeline accuracy number.

## Relationship To Library-Native Evaluation

For current public table reconstruction evaluation, prefer the library command:

```bash
tabulus evaluate-table-reconstruction \
  --gold /path/to/gold.csv \
  --prediction /path/to/prediction.csv \
  --metric rms \
  --out /path/to/evaluation.json
```

The native implementation is dataset-agnostic: it compares one gold CSV with
one prediction CSV and writes a JSON result only when `--out` is provided.

The retained table scripts in this directory are useful when reproducing older
research harnesses, but they may expect historical folder names and may write
result JSON files into dataset folders.

## Reference Extraction Utilities

The retained reference-extraction scripts compare predicted reference lists or
raw extracted bibliography text against curated gold references. Depending on
the script, matching is exact or similarity-based, and the output can include
true positives, false positives, false negatives, precision, recall, and F1.

Some historical outputs use the word `accuracy` for a found-reference ratio.
When documenting or reporting those results, describe that quantity as a
reference recovery rate unless the evaluation has a conventional classification
denominator with suitable negative examples.

These scripts evaluate bibliography/reference extraction only. They do not
evaluate Stage 5 table-cell-to-bibliography matching, Stage 6 scholarly identity
resolution, DOI validation, or planned Stage 7 exports.

## DePlot Material

`evaluation/deplot/` contains retained DePlot metric code used by older
research scripts. The current public Tabulus RMS implementation adapts the
DePlot table-datapoint metric inside `src/tabulus/evaluation/rms.py` so normal
library users do not need to import from this retained directory.

## Notes

- Do not commit generated evaluation outputs unless a repository maintainer
  explicitly decides they are documentation artifacts.
- Inspect script arguments before running retained utilities; several scripts
  write outputs next to the datasets they evaluate.
- Keep production artifacts read-only during evaluation.
- Keep reference-table classification, table reconstruction, bibliography
  extraction, reference matching, and scholarly resolution diagnostics separate.
