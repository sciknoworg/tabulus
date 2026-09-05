# Single PDF Local Workflow

This workflow is the development target for running one paper without Docker.

The current implemented local profiling command is:

```powershell
tabulus profile --pdf "C:\papers\INPUT.pdf" --backend pipeline
```

When `--out` is omitted, Tabulus writes to:

```text
C:\papers\tabulus-output\mineru\pipeline\
```

Use `--out` only when an explicit output directory is needed. MinerU keeps its native output hierarchy below the profiling directory.

After successful profiling, Tabulus automatically exports canonical MinerU table crops to:

```text
C:\papers\tabulus-output\table-crops\INPUT\
  tables_index.json
  images\
```

Use `tabulus export-table-crops` only when regenerating the normalized handoff from an existing MinerU output without rerunning MinerU.

The current implemented local table-reconstruction command runs one selected registered adapter. For example:

```powershell
tabulus reconstruct-tables `
  --crops "C:\papers\tabulus-output\table-crops\INPUT" `
  --adapter paddleocr-vl `
  --device cpu
```

If `--out` is omitted, Tabulus writes:

```text
C:\papers\tabulus-output\table-crops\INPUT\reconstructions\paddleocr-vl\
  native\
  parsed\
  predictions\
  batch_summary.json
```

The current standalone shape is:

```powershell
tabulus profile --pdf C:\papers\INPUT.pdf --backend pipeline
tabulus reconstruct-tables --crops C:\papers\tabulus-output\table-crops\INPUT --adapter paddleocr-vl --device cpu
```

Bibliography extraction is implemented through the Python API and requires a
running GROBID HTTP service:

```python
from pathlib import Path

from tabulus.bibliography.pipeline import extract_bibliography_artifact

extract_bibliography_artifact(
    Path("INPUT.pdf"),
    Path("OUTPUT_DIRECTORY"),
    grobid_url="http://localhost:8070",
)
```

The future complete command should remain under the same installed `tabulus` entry point:

```powershell
tabulus run --pdf C:\papers\INPUT.pdf --runs-root C:\runs
```

`tabulus run` is not implemented yet. The new library has registered
table-reconstruction adapters, the `tabulus reconstruct-tables` batch CLI, and
the GROBID-backed bibliography extraction library API. Stage 5 reference
matching writes `references/reference_matches.json` without modifying
prediction CSVs. DOI resolution, final resolved CSV export, and full run
reporting are not yet implemented.
