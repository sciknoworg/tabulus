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

Use `--out` only when an explicit output directory is needed. MinerU keeps its
native output hierarchy below the profiling directory.

After successful profiling, Tabulus automatically exports canonical MinerU
table crops to:

```text
C:\papers\tabulus-output\table-crops\INPUT\
  tables_index.json
  images\
```

Use `tabulus export-table-crops` only when regenerating the normalized handoff
from an existing MinerU output without rerunning MinerU.

The current implemented local table-reconstruction command runs one selected
registered adapter. For example:

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

Bibliography extraction requires a running GROBID HTTP service:

```powershell
tabulus extract-bibliography `
  --pdf C:\papers\INPUT.pdf `
  --out C:\runs\INPUT `
  --grobid-url http://localhost:8070
```

Reference-table classification and reference matching are also standalone
commands:

```powershell
tabulus classify-reference-tables `
  --reconstruction C:\papers\tabulus-output\table-crops\INPUT\reconstructions\paddleocr-vl

tabulus match-references `
  --selected C:\papers\tabulus-output\table-crops\INPUT\reconstructions\paddleocr-vl\selected_reference_tables.json `
  --bibliography C:\runs\INPUT\references\bibliography.json
```

The future complete command should remain under the same installed `tabulus`
entry point:

```powershell
tabulus run --pdf C:\papers\INPUT.pdf --runs-root C:\runs
```

`tabulus run` is not implemented yet. The new library has registered
table-reconstruction adapters, the `tabulus reconstruct-tables` batch CLI,
reference-table classification, the GROBID-backed
`tabulus extract-bibliography` CLI, and deterministic Stage 5 reference
matching. Stage 6 scholarly reference resolution, Stage 7 resolved export, and
full run reporting remain unimplemented in this repository.
