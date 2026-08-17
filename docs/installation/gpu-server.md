# GPU Server Installation

This page documents the supported Tabulus + MinerU GPU installation and profiling workflow using MinerU's `hybrid-engine` backend.

A GPU is not required for all Tabulus use. Windows and CPU-only machines can use the `pipeline` backend documented in `installation/windows-cpu`.

The GPU workflow is:

```text
NVIDIA GPU
     |
     v
Create tabulus-mineru Conda environment
     |
     v
Install Tabulus
     |
     v
Install MinerU 3.4.5
     |
     v
Verify PyTorch CUDA access
     |
     v
tabulus profile --backend hybrid-engine
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

## 2. Verify The NVIDIA GPU

Confirm that NVIDIA GPUs are available:

```bash
nvidia-smi
```

On the tested system, four NVIDIA L40S GPUs with approximately 46 GB VRAM each are available.

For initial testing, restrict execution to one GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 3. Clone / Enter The Tabulus Repository

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

## 6. Install MinerU

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

## 8. Run GPU Profiling

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

## 9. Inspect Profiling Output

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
