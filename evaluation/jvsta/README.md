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

## Profile PDF corpora under the same harness

Publication profiling runs wrap the real `tabulus profile` CLI and write MinerU output plus canonical table crops into an isolated run directory:

```bash
python -m evaluation.jvsta.run_profiling_experiment \
  --corpus ald \
  --pdf-folder "$HOME/ald-papers" \
  --backend hybrid-engine \
  --run-root "$HOME/jvsta-experiments" \
  --expected-commit <TABULUS_GIT_SHA>
```

Profiling metadata records PDF counts, page counts when `pdfinfo` is available, profiling wall time, host RSS, Slurm/GPU identity, tables detected, and canonical crops saved. `pipeline` runs do not sample GPU memory; `hybrid-engine` runs do.

## Slurm matrix launcher

`evaluation/jvsta/slurm/submit_matrix.py` renders `sbatch` commands from a CSV manifest. It is deliberately dry-run by default:

```bash
python -m evaluation.jvsta.slurm.submit_matrix \
  --manifest evaluation/jvsta/manifests/profiling.example.csv
```

Review the rendered commands, then submit explicitly:

```bash
python -m evaluation.jvsta.slurm.submit_matrix \
  --manifest evaluation/jvsta/manifests/profiling.example.csv \
  --submit
```

The launcher pins every job to the current clean Tabulus commit unless `--expected-commit` is supplied. The bundled profiling example uses no GPU GRES for the CPU `pipeline` backend and `gpu:l40s:1` for `hybrid-engine`. The reconstruction example uses one L40S GPU per job and leaves the canonical crop paths as `FROZEN-CROPS` placeholders until a paper crop set is explicitly frozen.

On the TIB cluster, Slurm may report a physical allocation such as `SLURM_JOB_GPUS=2` while exposing that isolated device to the process as `CUDA_VISIBLE_DEVICES=0`. Run metadata and summary CSVs preserve both identities plus the visible GPU UUID.
