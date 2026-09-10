# End-To-End Run

An end-to-end runner should only orchestrate stable standalone components.

The rebuilt library now provides standalone CLI commands for profiling,
table-crop export, batch table reconstruction, bibliography extraction,
reference matching, Step 6 paper-level scholarly reference resolution, and
Step 7 resolved CSV export. It does not yet provide a complete end-to-end
`tabulus run` command. This page describes the target workflow shape.

## Rule

Do not hide component failures. The end-to-end runner should stop at the failed step, write a run report, and leave all intermediate outputs inspectable.

## Current Standalone Commands

```bash
tabulus profile --pdf /data/papers/INPUT.pdf --backend hybrid-engine

tabulus export-table-crops \
  --mineru-root /data/papers/tabulus-output/mineru/hybrid-engine/<document>/<run> \
  --out /data/papers/tabulus-output/table-crops/INPUT

tabulus reconstruct-tables \
  --crops /data/papers/tabulus-output/table-crops/INPUT \
  --adapter paddleocr-vl \
  --device gpu:0

tabulus classify-reference-tables \
  --reconstruction /data/papers/tabulus-output/table-crops/INPUT/reconstructions/paddleocr-vl

tabulus extract-bibliography \
  --pdf /data/papers/INPUT.pdf \
  --out /data/runs/INPUT \
  --grobid-url http://localhost:8070

tabulus match-references \
  --selected /data/papers/tabulus-output/table-crops/INPUT/reconstructions/paddleocr-vl/selected_reference_tables.json \
  --bibliography /data/runs/INPUT/references/bibliography.json

tabulus resolve-references \
  --bibliography /data/runs/INPUT/references/bibliography.json \
  --reference-matches /data/papers/tabulus-output/table-crops/INPUT/reconstructions/paddleocr-vl/references/reference_matches.json \
  --out /data/runs/INPUT

tabulus export-resolved-csv \
  --reference-matches /data/papers/tabulus-output/table-crops/INPUT/reconstructions/paddleocr-vl/references/reference_matches.json \
  --reference-resolution /data/runs/INPUT/references/reference_resolution.json
```

`tabulus profile` already exports the canonical table-crop handoff by default, so `export-table-crops` is mainly for regenerating the handoff from an existing MinerU run.

## Future Target Command

```bash
tabulus run --pdf /data/papers/INPUT.pdf --runs-root /data/runs
```

`tabulus run` is future work. It should orchestrate stable standalone components rather than embed their implementations.

## Output

The output should be the same run directory that individual component commands produce.

```text
runs/<paper>/
  profiling/                         external-tool native outputs
  tables/crops/                      canonical table-crop handoff
  tables/reconstructions/<adapter>/  native, parsed, and prediction artifacts
  references/                        bibliography, reference matches, and reference resolution
  resolved_reference_tables/         Step 7 resolved physical CSV files
  evaluation/                        metrics and comparison reports
  report/                            run summary and QA bundle
```

Production components should preserve stable intermediate artifacts so evaluation can be reproduced. Evaluation reads explicit scored artifacts such as prediction CSV files or reference manifests, writes separate metrics, and never mutates the production artifacts it scores.
