# Debugging A Failed Step

Failures should be debugged at the component boundary.

## Debugging Pattern

1. Open the run report.
2. Identify the first failed step.
3. Inspect that step's input contract.
4. Inspect that step's raw output and logs.
5. Rerun only that step with the same run directory.

## Example

If table reconstruction fails, do not rerun PDF profiling. Rerun only:

```bash
tabulus reconstruct-tables \
  --crops /data/papers/tabulus-output/table-crops/P51 \
  --adapter paddleocr-vl \
  --device gpu:0
```
