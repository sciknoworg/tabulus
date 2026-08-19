# End-To-End Run

An end-to-end runner should only orchestrate stable standalone components.

The rebuilt library does not yet provide a complete end-to-end command. This page describes the target workflow shape.

## Rule

Do not hide component failures. The end-to-end runner should stop at the failed step, write a run report, and leave all intermediate outputs inspectable.

## Target Command

```bash
python -m tabulus_pipeline.run_all --pdf /data/papers/P51.pdf --runs-root /data/runs
```

## Output

The output should be the same run directory that individual component commands produce.

```text
runs/<paper>/
  profiling/                         external-tool native outputs
  tables/crops/                      canonical table-crop handoff
  tables/reconstructions/<adapter>/  native, normalized, and prediction artifacts
  references/                        bibliography and reference matches
  resolved_reference_tables/         DOI-enriched CSV files
  evaluation/                        metrics and comparison reports
  report/                            run summary and QA bundle
```

Production components should preserve stable intermediate artifacts so evaluation can be reproduced. Evaluation reads prediction CSV files and other production artifacts, writes metrics under `evaluation/`, and never mutates the production artifacts it scores.
