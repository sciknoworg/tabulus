# Evaluation

Evaluation pages describe how to score Tabulus artifacts without mutating the
production pipeline outputs. The current library exposes pair-level evaluation
APIs and public CLI commands for Steps 1--6.

Each Step has its own artifact, gold source, denominator, and metric. These
scores should not be collapsed into one pipeline accuracy number.

```{toctree}
:maxdepth: 1

overview
table-localization-quality
table-extraction-quality
reference-table-classification-quality
bibliography-quality
reference-matching-quality
reference-resolution-quality
```
