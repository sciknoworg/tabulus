# JVSTA experiment harness

This directory contains repository-local tooling for the ALD/ASD corpus-scale JVSTA experiments. It wraps the real Tabulus CLI; it does not implement an alternative reconstruction path.

For one reconstruction run it records:

- corpus, adapter, exact commands, Python/Conda environment and Tabulus Git commit;
- Slurm partition/job/node/GPU environment;
- GPU model, total VRAM and sampled peak device memory;
- full reconstruction wall time and GNU `time -v` maximum host RSS when available;
- Tabulus batch elapsed time, per-table timings, coverage and parsing outcomes;
- failure taxonomy;
- output-complexity summaries;
- adapter/model provenance from native Tabulus artifacts;
- downstream reference-table classification counts when classification is enabled.

Publication runs refuse tracked Git modifications by default. Use `--expected-commit` to pin a campaign explicitly.

## Run one corpus/adapter experiment

Run from the repository root inside the adapter's tested environment, preferably as a Slurm job with one experiment per allocated GPU:

```bash
python -m evaluation.jvsta.run_experiment \
  --corpus ald \
  --crops-folder "$HOME/ald-papers/tabulus-output/table-crops" \
  --adapter internvl3-5-8b \
  --device gpu:0 \
  --run-root "$HOME/jvsta-experiments" \
  --expected-commit <TABULUS_GIT_SHA>
```

Each invocation creates a unique directory below:

```text
<run-root>/<corpus>/<adapter>/<run-id>/
```

The directory contains `run_metadata.json`, GPU samples, stdout/stderr logs, resource logs, and an isolated `reconstruction/` tree. Existing development reconstructions are never overwritten.

`peak_gpu_memory_mib` is sampled from `nvidia-smi memory.used` for the best-effort allocated GPU selection. For publication timing, use one experiment per physical GPU so device-wide memory is attributable to that job.

## Summarize completed runs

```bash
python -m evaluation.jvsta.summarize_experiments \
  --run-root "$HOME/jvsta-experiments" \
  --out "$HOME/jvsta-experiments/jvsta_runs.csv"
```

The CSV contains coverage, parsing, runtime, resource, robustness, downstream, output-complexity, and reproducibility fields suitable for later paper tables and plots.
