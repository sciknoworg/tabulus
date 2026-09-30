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

## Step 6 Scholarly Identity Resolution

Step 6 evaluation is separate from Step 5 reference matching. Step 5 establishes
which bibliography entry was cited; it does not establish the canonical
scholarly-work identity. Step 6 validation status is therefore not accuracy by
itself, and `validated_with_doi` is an output status rather than an accuracy
label.

The controlled Step 6 component workflow uses clean one-to-one alignments
between controlled Step 5 bibliography indices and Step 4/GROBID bibliography
entries. Derived inputs live under the literal run tree:

```text
runs/stage6/controlled-p251-p252-final/<paper>/input_one_to_one/
```

Those inputs are used to assess resolver behavior while keeping upstream
bibliography-segmentation split, merge, and unmatched cases separate from the
clean Step 6 component population. The one-to-one controlled subset must not be
presented as a full end-to-end pipeline population.

Because Step 6 depends on live Crossref, CORE, network/provider behavior, and
LLM-assisted adjudication, controlled consistency checks use two fresh complete
runs per paper (`run_01` and `run_02`) and compare individual-reference
outcomes. Relevant checks include final-status agreement, exact DOI agreement
for references validated in both runs, validated-identity agreement, retry-use
agreement, LLM-decision agreement where applicable, and the number and type of
references whose final outcome changes.

Observed Step 6 runtime should be reported as end-to-end wall-clock runtime,
not pure computation time, because it includes remote Crossref, CORE, and LLM
provider latency. Where instrumented, record processed references,
seconds/reference, number of LLM-processed entries, bounded retries, and
optionally CPU time and maximum resident memory.

Step 6 scholarly-identity accuracy requires independent identity gold. Without
that gold, validation yield, DOI yield, rejection counts, and run-to-run
agreement are resolver diagnostics rather than precision, recall, F1, DOI
accuracy, or scholarly-identity accuracy.
