# Evaluation Overview

Evaluation is operationally separate from the production pipeline.

The `evaluation/` folder contains scripts, plots, and metrics used to compare OCR models, table extraction quality, bibliography extraction quality, and reference matching behavior.

Production components should preserve stable intermediate artifacts so evaluation can be reproduced, but evaluation scripts should never mutate production artifacts. For table reconstruction, the scored artifact is the prediction CSV, not the later DOI-enriched resolved CSV.

```text
ground-truth CSV
        ^
        |
RMS / DePlot evaluation
        |
prediction CSV
        ^
        |
normalized reconstruction
```

The main evaluation levels are:

- table reconstruction quality against ground-truth CSV files
- structural subclass evaluation by table type or difficulty class
- bibliography extraction quality against curated reference lists
- reference extraction and matching quality
- runtime and hardware context for each stage

Resolved CSV files are useful for end-user inspection and DOI-enrichment QA, but they are not the artifact used to measure raw OCR/table reconstruction quality.
