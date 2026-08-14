# End-To-End Run

An end-to-end runner should only orchestrate stable standalone components.

## Rule

Do not hide component failures. The end-to-end runner should stop at the failed step, write a run report, and leave all intermediate outputs inspectable.

## Target Command

```bash
python -m tabulus_pipeline.run_all --pdf /data/papers/P51.pdf --runs-root /data/runs
```

## Output

The output should be the same run directory that individual component commands produce.
