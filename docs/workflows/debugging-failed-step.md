# Debugging A Failed Step

Failures should be debugged at the component boundary.

## Debugging Pattern

1. Open the run report.
2. Identify the first failed step.
3. Inspect that step's input contract.
4. Inspect that step's raw output and logs.
5. Rerun only that step with the same run directory.

## Example

If table OCR fails, do not rerun PDF ingestion or page rendering. Rerun only:

```bash
python -m tabulus_pipeline.ocr_tables --run /data/runs/P51 --adapter paddleocr_vl
```
