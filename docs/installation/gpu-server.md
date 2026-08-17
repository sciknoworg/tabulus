# GPU Server Installation

This page documents the supported Tabulus + MinerU GPU installation and profiling workflow using MinerU's `hybrid-engine` backend.

A GPU is not required for all Tabulus use. Windows and CPU-only machines can use the `pipeline` backend documented in `installation/windows-cpu`.

The GPU workflow begins with a Slurm allocation and then runs the Tabulus + MinerU software stack inside that allocation:

```text
Connect to server
     |
     v
Login node
     |
     v
Request Slurm resources
     |
     v
Allocated compute node
     |
     v
Verify GPU with nvidia-smi
     |
     v
Activate tabulus-mineru
     |
     v
Verify PyTorch CUDA
     |
     v
Run Tabulus / MinerU
     |
     v
MinerU profiling output
```

Do not rely on the repository-level `requirements.txt` to install MinerU GPU dependencies. The root requirements are the lightweight Tabulus library development contract; MinerU should be installed in its own environment.

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
srun --partition=p_12G \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=02:00:00 \
  --pty bash
```

The options mean:

- `--partition=p_12G`: Slurm partition to submit to.
- `--gres=gpu:1`: request one GPU.
- `--cpus-per-task=8`: request eight CPU cores.
- `--mem=32G`: request 32 GB host/system RAM.
- `--time=02:00:00`: maximum allocation time of two hours.
- `--pty bash`: start an interactive shell on the allocated node.

`--mem=32G` requests system RAM, not GPU memory. The GPU has its own VRAM; requesting a GPU with `--gres=gpu:1` gives the job access to an allocated GPU.

After the interactive allocation starts, verify that you are on the allocated compute node and that the assigned GPU is visible:

```bash
hostname
nvidia-smi
```

This is an interactive allocation. Batch execution is a separate Slurm mode using `sbatch`; it should be documented separately if a validated Tabulus batch workflow is added.

Keep Slurm resource allocation separate from environment installation. Slurm controls where and how the computation runs; Conda controls which Python software and dependencies run inside that allocation.

## 3. Verify The NVIDIA GPU

Confirm that NVIDIA GPUs are available:

```bash
nvidia-smi
```

On the tested system, four NVIDIA L40S GPUs with approximately 46 GB VRAM each are available.

For initial testing, restrict execution to one GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 4. Clone / Enter The Tabulus Repository

Clone Tabulus and enter the repository:

```bash
git clone https://github.com/sciknoworg/tabulus.git
cd tabulus
```

Define the working directory:

```bash
export TABULUS_ROOT="$HOME/tabulus"
export WORK="$TABULUS_ROOT/work"
```

Create an input directory for PDFs:

```bash
mkdir -p "$WORK/input"
```

Place the PDF to profile under `$WORK/input`.

## 5. Create The tabulus-mineru Conda Environment

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

## 6. Install Tabulus

Install Tabulus in editable mode from the repository checkout:

```bash
python -m pip install -e .
```

Verify the Python executable:

```bash
python --version
which python
```

Expected path shape:

```text
~/miniconda3/envs/tabulus-mineru/bin/python
```

## 7. Install MinerU

Install MinerU in the `tabulus-mineru` environment:

```bash
python -m pip install "mineru[all]==3.4.5"
```

Then verify:

```bash
mineru --version
```

The tested version was:

```text
MinerU 3.4.5
```

## 8. Verify PyTorch CUDA Access

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

## 9. Run GPU Profiling

Run MinerU through the Tabulus CLI with the GPU backend:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus profile \
  --pdf "$WORK/input/Puurunen - February 2005.pdf" \
  --backend hybrid-engine \
  --effort high \
  --method auto
```

Do not pass `--out` unless you intentionally want to override the output root. When `--out` is omitted, Tabulus uses its current automatic profiling output convention:

```text
<PDF directory>/
└── tabulus-output/
    └── mineru/
        └── hybrid-engine/
```

This directory is the profiler/backend output root passed to MinerU. MinerU then creates its native document and method hierarchy underneath that root. Do not flatten or rename MinerU-native output files.

If `hybrid-engine` is requested but GPU requirements are not satisfied, Tabulus reports the reason and falls back to the CPU-compatible `pipeline` backend. In that case, the automatic output root uses the resolved backend:

```text
<PDF directory>/
└── tabulus-output/
    └── mineru/
        └── pipeline/
```

## 10. Inspect Profiling Output

For the example command above, the automatic GPU output root is:

```text
$WORK/input/tabulus-output/mineru/hybrid-engine/
```

Inspect the generated MinerU-native hierarchy after the run:

```bash
find "$WORK/input/tabulus-output/mineru/hybrid-engine" -maxdepth 4 -type f | sort
```

The exact MinerU-native document and method subdirectory produced by `hybrid-engine` should be confirmed from the fresh GPU run before being documented as a fixed layout.

CPU-only profiling is documented separately in `installation/windows-cpu`.
