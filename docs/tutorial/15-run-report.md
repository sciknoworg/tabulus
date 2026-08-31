# Planned Stage: Run Report And QA Bundle

## Goal

Make the run inspectable after processing, including failures.

## Input

All previous step outputs.

## Output

`report/run_report.json` and optional human-readable `report/summary.md`.

## Module Contract

```json
{
  "run_id": "P51",
  "status": "completed_with_warnings",
  "steps": [
    {
      "name": "pdf_ingestion",
      "status": "success",
      "duration_seconds": 0.2
    }
  ],
  "warnings": [],
  "errors": []
}
```

## Default Implementation

This stage is planned for the rebuilt `src/tabulus` library. There is not yet a complete `tabulus run` command or run-report / QA-bundle writer.

## Verification

The step succeeds when a human can tell which module ran, what it produced, and where the run failed if it failed.
