# Single PDF Local Workflow

This workflow is the development target for running one paper without Docker.

The current implemented local profiling command is:

```powershell
tabulus profile --pdf "C:\papers\P51.pdf" --backend pipeline
```

When `--out` is omitted, Tabulus writes to:

```text
C:\papers\tabulus-output\mineru\pipeline\
```

Use `--out` only when an explicit output directory is needed. MinerU keeps its native output hierarchy below the profiling directory.

After successful profiling, Tabulus automatically exports canonical MinerU table crops to:

```text
C:\papers\tabulus-output\table-crops\P51\
  tables_index.json
  images\
```

Use `tabulus export-table-crops` only when regenerating the normalized handoff from an existing MinerU output without rerunning MinerU.

The current implemented local table-reconstruction command is:

```powershell
tabulus reconstruct-tables `
  --crops "C:\papers\tabulus-output\table-crops\P51" `
  --adapter paddleocr-vl `
  --device cpu
```

If `--out` is omitted, Tabulus writes:

```text
C:\papers\tabulus-output\table-crops\P51\reconstructions\paddleocr-vl\
  native\
  parsed\
  predictions\
  batch_summary.json
```

The current standalone shape is:

```powershell
tabulus profile --pdf C:\papers\P51.pdf --backend pipeline
tabulus reconstruct-tables --crops C:\papers\tabulus-output\table-crops\P51 --adapter paddleocr-vl --device cpu
```

The future complete command should remain under the same installed `tabulus` entry point:

```powershell
tabulus run --pdf C:\papers\P51.pdf --runs-root C:\runs
```

`tabulus run` is not implemented yet. The new library has a PaddleOCR-VL table-reconstruction adapter and the `tabulus reconstruct-tables` batch CLI, but bibliography extraction, reference matching, DOI resolution, final resolved CSV export, and full run reporting are not yet implemented.
