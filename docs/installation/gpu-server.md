# GPU Server Installation

This page documents the supported Tabulus + MinerU GPU installation and profiling workflow using MinerU's `hybrid-engine` backend.

A GPU is not required for all Tabulus use. Windows and CPU-only machines can use the `pipeline` backend documented in `installation/windows-cpu`.

The GPU workflow begins with a Slurm allocation and then runs the Tabulus + MinerU software stack inside that allocation:

```text
SSH to server
     |
     v
login node
     |
     v
request Slurm resources
     |
     v
allocated compute node
     |
     +-- hostname
     +-- nvidia-smi
     |
     v
cd $TABULUS_ROOT
     |
     v
create/activate tabulus-mineru
     |
     v
install Tabulus + MinerU
     |
     v
verify PyTorch CUDA access
     |
     v
tabulus profile --pdf "$PAPERS/..."
     |
     v
$PAPERS/tabulus-output/mineru/hybrid-engine/...
```

## 1. Tested Environment

The verified setup uses:

- Linux GPU server
- NVIDIA L40S GPUs
- NVIDIA driver 595.71.05
- Conda or Miniconda
- Python 3.12
- Tabulus installed from the repository checkout
- MinerU 3.4.5

## 2. Request GPU Compute Resources

Connecting to the GPU server over SSH usually places you on a login node. Logging into the server does not by itself allocate a GPU, CPU cores, or RAM for computation. Resource allocation is handled separately by the Slurm scheduler.

The distinction is:

```text
SSH connection to GPU server
         |
         v
login node
         |
         | request resources from Slurm
         v
allocated compute node
         |
         v
GPU / CPU / RAM available to the job
```

For the interactive workflow used during Tabulus testing, request an interactive allocation:

```bash
srun --partition=p_48G \
  --nodelist=gpu-l40s-02 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=12:00:00 \
  --pty bash
```

The options mean:

- `--partition=p_48G`: Slurm partition to submit to.
- `--nodelist=gpu-l40s-02`: request the specific `gpu-l40s-02` node for this validated workflow.
- `--gres=gpu:1`: request one GPU.
- `--cpus-per-task=8`: request eight CPU cores.
- `--mem=32G`: request 32 GB host/system RAM.
- `--time=12:00:00`: maximum allocation time of 12 hours.
- `--pty bash`: start an interactive shell on the allocated node.

`--mem=32G` requests system RAM, not GPU memory. The GPU has its own VRAM; requesting a GPU with `--gres=gpu:1` gives the job access to an allocated GPU.

`--nodelist=gpu-l40s-02` is not inherently required by Slurm, but it is currently used in the documented validation command to guarantee that the job runs on an L40S node.

This has a trade-off:

- With `--nodelist=gpu-l40s-02`, the run is reproducible on that exact node.
- If `gpu-l40s-02` is busy, the job may wait even if another suitable L40S node is available.
- Using only `--partition=p_48G --gres=gpu:1` does not guarantee an L40S on this cluster because the partition contains multiple GPU types.

A better long-term command would request any L40S using a model-specific GRES or Slurm feature/constraint, if the cluster exposes one. For example, the final command might use something conceptually like `--gres=gpu:l40s:1` or `--constraint=l40s`, but do not treat either form as valid until the cluster configuration has been verified.

After the interactive allocation starts, verify that you are on the allocated compute node and that the assigned GPU is visible:

```bash
hostname
nvidia-smi
```

Optionally inspect available nodes, partitions, GPU resources, memory, and node state:

```bash
sinfo -N -o "%N %P %G %m %t"
```

For initial testing, restrict software execution to one GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

This is an interactive allocation. Batch execution is a separate Slurm mode using `sbatch`; it should be documented separately if a validated Tabulus batch workflow is added.

Keep Slurm resource allocation separate from environment installation. Slurm controls where and how the computation runs; Conda controls which Python software and dependencies run inside that allocation.

## 3. Clone / Enter The Tabulus Repository

Clone Tabulus and enter the repository:

```bash
git clone https://github.com/sciknoworg/tabulus.git
cd tabulus
```

Keep the source repository and the PDF collection separate. The intended layout is:

```text
$HOME/
├── tabulus/
│   ├── src/
│   ├── docs/
│   ├── tests/
│   └── ...
│
└── <papers-folder>/
    ├── Puurunen - February 2005.pdf
    └── tabulus-output/
        └── mineru/
            └── hybrid-engine/
```

Define the source and data locations separately:

```bash
export TABULUS_ROOT="$HOME/tabulus"
export PAPERS="$HOME/<papers-folder>"
```

Verify that the PDF collection is available:

```bash
cd "$TABULUS_ROOT"
ls "$PAPERS"
```

In this workflow, source code lives under `$TABULUS_ROOT`. Input PDFs and generated profiling output live under `$PAPERS`.

## 4. Create The tabulus-mineru Conda Environment

Deactivate the current environment if necessary:

```bash
conda deactivate
```

Create a Python 3.12 environment:

```bash
conda create -n tabulus-mineru python=3.12 -y
```

Activate it:

```bash
conda activate tabulus-mineru
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

## 5. Install Tabulus

For normal library use, install Tabulus in editable mode from the repository checkout:

```bash
python -m pip install -e .
```

For development and testing, install the development extra instead:

```bash
python -m pip install -e ".[dev]"
```

The `dev` extra installs development/test dependencies, including `pytest`. `pytest` is not installed by the normal `python -m pip install -e .` installation.

You do not need to run `python -m pip install -e .` first if you use `python -m pip install -e ".[dev]"`.

The distinction is:

```text
Normal use
    |
    +-- python -m pip install -e .
    |
    +-- run Tabulus

Development / validation
    |
    +-- python -m pip install -e ".[dev]"
    |
    +-- python -m pytest -v
    |
    +-- run Tabulus
```

Verify the Python executable and CLI:

```bash
python --version
which python
tabulus --version
```

Expected path shape:

```text
~/miniconda3/envs/tabulus-mineru/bin/python
```

## 6. Install MinerU

Install MinerU in the `tabulus-mineru` environment:

```bash
python -m pip install "mineru[all]==3.4.5"
```

For the MinerU options and output artifacts used by Tabulus, see {doc}`../external-tools/mineru`.

Then verify:

```bash
mineru --version
```

The tested version was:

```text
MinerU 3.4.5
```

## 7. Verify PyTorch CUDA Access

Verify CUDA access from the same Conda environment that will execute MinerU:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

This verifies that PyTorch inside the `tabulus-mineru` Conda environment can access the NVIDIA GPU, which is the relevant GPU check for MinerU.

In the tested environment this resolved to one visible NVIDIA L40S GPU when `CUDA_VISIBLE_DEVICES=0` was set.

## 8. Run The Tabulus Test Suite

This section applies when validating a development checkout, modifying Tabulus source code, or preparing changes for commit. Running the tests is not required merely to use an already validated Tabulus installation.

Run the test suite with:

```bash
python -m pytest -v
```

Linux validation after the MinerU native-run-directory runner fix collected 22 tests and all 22 passed. That run included the regression test `test_run_mineru_returns_native_hybrid_run_dir`, which verifies that Tabulus returns the native `hybrid_auto/` directory for the validated hybrid run and does not create an artificial `auto/` directory.

This requires Tabulus to have been installed with:

```bash
python -m pip install -e ".[dev]"
```

If you already selected the development/testing installation in Section 5, no additional install command is needed.

## 9. Run GPU Profiling

Run MinerU through the Tabulus CLI with the GPU backend:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus profile \
  --pdf "$PAPERS/Puurunen - February 2005.pdf" \
  --backend hybrid-engine \
  --effort high \
  --method auto
```

Do not pass `--out` unless you intentionally want to override the output root. When `--out` is omitted, Tabulus places the automatic output relative to the PDF's parent directory:

```text
$PAPERS/
├── Puurunen - February 2005.pdf
└── tabulus-output/
    └── mineru/
        └── hybrid-engine/
            └── ...
```

This directory is the profiler/backend output root passed to MinerU. MinerU then creates its native document/run hierarchy underneath that root. Do not flatten or rename MinerU-native output files.

If `hybrid-engine` is requested but GPU requirements are not satisfied, Tabulus reports the reason and falls back to the CPU-compatible `pipeline` backend. In that case, the automatic output root uses the resolved backend:

```text
$PAPERS/
└── tabulus-output/
    └── mineru/
        └── pipeline/
```

## 10. Inspect Profiling Output

For the example command above, the automatic GPU output root is:

```text
$PAPERS/tabulus-output/mineru/hybrid-engine/
```

Inspect the generated MinerU-native hierarchy after the run:

```bash
find "$PAPERS/tabulus-output/mineru/hybrid-engine" -maxdepth 4 -type f | sort
```

For the validated MinerU 3.4.5 Linux GPU run on NVIDIA L40S with:

```text
--backend hybrid-engine
--method auto
--effort high
```

MinerU generated the native run directory `hybrid_auto`:

```text
$PAPERS/
├── <document>.pdf
└── tabulus-output/
    └── mineru/
        └── hybrid-engine/
            └── <document>/
                └── hybrid_auto/
                    ├── images/
                    ├── <document>_content_list.json
                    ├── <document>_content_list_v2.json
                    ├── <document>_layout.pdf
                    ├── <document>_middle.json
                    ├── <document>_model.json
                    ├── <document>_origin.pdf
                    ├── <document>.md
                    ├── mineru_stdout.log
                    ├── mineru_stderr.log
                    └── tabulus_run.txt
```

`hybrid_auto` is MinerU's native directory name for the validated `hybrid-engine` + `auto` run. It is not a Tabulus-created directory and should not be treated as a guarantee for future MinerU versions.

A successful CLI run should report the discovered native run directory, for example:

```text
PDF profiling completed: .../<document>/hybrid_auto
```

CPU-only profiling is documented separately in `installation/windows-cpu`.
