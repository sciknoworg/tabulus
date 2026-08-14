# Single PDF Local Workflow

This workflow is the development target for running one paper without Docker.

```powershell
python -m tabulus_pipeline.profile_pdf --pdf C:\papers\P51.pdf --runs-root C:\runs --adapter mineru
python -m tabulus_pipeline.ocr_tables --run C:\runs\P51 --adapter paddleocr_vl
python -m tabulus_pipeline.classify_reference_tables --run C:\runs\P51
python -m tabulus_pipeline.extract_bibliography --run C:\runs\P51 --adapter grobid
python -m tabulus_pipeline.match_references --run C:\runs\P51
python -m tabulus_pipeline.export_csv --run C:\runs\P51
python -m tabulus_pipeline.report_run --run C:\runs\P51
```

These commands are aspirational module boundaries. The current code must be refactored toward them.
