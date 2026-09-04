# End-To-End Run

An end-to-end runner should only orchestrate stable standalone components.

The rebuilt library now provides standalone CLI commands for profiling,
table-crop export, and batch table reconstruction, plus a Python API for
GROBID-backed bibliography extraction. It does not yet provide a complete
end-to-end `tabulus run` command. This page describes the target workflow
shape.

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
```

`tabulus profile` already exports the canonical table-crop handoff by default, so `export-table-crops` is mainly for regenerating the handoff from an existing MinerU run.

Bibliography extraction currently uses the Python API:

```python
from pathlib import Path

from tabulus.bibliography.pipeline import extract_bibliography_artifact

extract_bibliography_artifact(
    Path("/data/papers/INPUT.pdf"),
    Path("/data/runs/INPUT"),
    grobid_url="http://localhost:8070",
)
```

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
  references/                        bibliography and reference matches
  resolved_reference_tables/         DOI-enriched CSV files
  evaluation/                        metrics and comparison reports
  report/                            run summary and QA bundle
```

Production components should preserve stable intermediate artifacts so evaluation can be reproduced. Evaluation reads prediction CSV files and other production artifacts, writes metrics under `evaluation/`, and never mutates the production artifacts it scores.
