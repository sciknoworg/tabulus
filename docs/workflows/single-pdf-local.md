# Single PDF Local Workflow

This workflow is the development target for running one paper without Docker.

The current implemented local profiling command is:

```powershell
tabulus profile --pdf "C:\papers\P51.pdf" --backend pipeline
```

When `--out` is omitted, Tabulus writes to:

```text
C:\papers\tabulus-output\P51\profiling\mineru\pipeline\
```

Use `--out` only when an explicit output directory is needed. MinerU keeps its native output hierarchy below the profiling directory.

The current implemented table-crop handoff command is:

```powershell
tabulus export-table-crops --mineru-root "C:\papers\tabulus-output\P51\profiling\mineru\pipeline" --out "C:\papers\tabulus-output\P51\table_crops"
```

The future end-to-end shape is:

```powershell
python -m tabulus_pipeline.profile_pdf --pdf C:\papers\P51.pdf --runs-root C:\runs --adapter mineru
python -m tabulus_pipeline.ocr_tables --run C:\runs\P51 --adapter paddleocr_vl
python -m tabulus_pipeline.classify_reference_tables --run C:\runs\P51
python -m tabulus_pipeline.extract_bibliography --run C:\runs\P51 --adapter grobid
python -m tabulus_pipeline.match_references --run C:\runs\P51
python -m tabulus_pipeline.export_csv --run C:\runs\P51
python -m tabulus_pipeline.report_run --run C:\runs\P51
```

These `tabulus_pipeline.*` commands are aspirational module boundaries. PaddleOCR-VL execution, bibliography extraction, reference matching, DOI resolution, CSV export, and full run reporting are not yet implemented in the new library.
