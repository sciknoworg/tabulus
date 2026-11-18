# Evaluation

Evaluation pages describe how to score Tabulus artifacts without mutating the
production pipeline outputs. The current library exposes evaluation APIs and
public CLI commands for Steps 1--7. Step 7 has separate physical-grid and
optional structural/ORKG evaluators.

**For Step 2 and physical Step 7, start here:**
{doc}`physical-grid-evaluation`.
Controlled reconstructions, aligned production reconstructions, and aligned
resolved exports use the same physical-grid scorer against curated Step 2
`gold.csv` tables. Batch reports return macro and micro scores together;
these are aggregation methods, not separate metrics or commands.

Other evaluation questions have their own gold sources and denominators.
These scores should not be collapsed into one pipeline accuracy number.

```{toctree}
:maxdepth: 1

overview
table-localization-quality
table-extraction-quality
physical-grid-evaluation
reference-table-classification-quality
bibliography-quality
reference-matching-quality
reference-resolution-quality
step7-quality
```
