# GPU Server Installation

Tabulus can be prepared for execution on an NVIDIA GPU server without Docker.

This setup is intended for the modular workflow where the web UI is not required. Intermediate outputs are written to folders so each processing stage can be inspected independently.

## Tested Environment

The verified setup uses:

- Linux GPU server
- NVIDIA L40S GPUs
- NVIDIA driver 595.71.05
- Apptainer 1.4.5
- Conda or Miniconda
- Python 3.12
- separate Conda environments for MinerU and PaddleOCR
- no Docker runtime

Do not use the repository-level `requirements.txt` for this GPU installation. It contains environment-specific and legacy dependencies and is not suitable as the primary HPC installation method.

## Verify The GPU

Confirm that NVIDIA GPUs are available:

```bash
nvidia-smi
```

On the tested system, four NVIDIA L40S GPUs with approximately 46 GB VRAM each are available.

For initial testing, restrict execution to one GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Verify Apptainer

Docker may not be installed on HPC systems. Tabulus was tested with Apptainer, which can execute Docker/OCI images.

Check the installation:

```bash
apptainer --version
```

The tested version is:

```text
apptainer version 1.4.5-3.el9
```

The `singularity` command may also point to Apptainer:

```bash
singularity --version
```

## Test NVIDIA GPU Passthrough

Before installing Tabulus dependencies, verify that a container can access the NVIDIA GPU:

```bash
apptainer exec --nv \
  docker://nvidia/cuda:12.6.3-base-ubuntu22.04 \
  nvidia-smi
```

The first execution downloads the OCI image and converts it to a SIF image.

Successful execution should show the GPUs from inside the container.

## Test PaddlePaddle GPU Compatibility

Tabulus uses PaddleOCR for table recognition. Before installing PaddleOCR, verify GPU compatibility with the PaddlePaddle GPU image:

```bash
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv \
  docker://paddlepaddle/paddle:3.3.0-gpu-cuda11.8-cudnn8.9 \
  python3 -c "import paddle; print('Paddle:', paddle.__version__); print('CUDA:', paddle.device.is_compiled_with_cuda()); print('GPUs:', paddle.device.cuda.device_count()); paddle.utils.run_check()"
```

The tested output was equivalent to:

```text
Paddle: 3.3.0
CUDA: True
GPUs: 1
PaddlePaddle works well on 1 GPU.
PaddlePaddle is installed successfully!
```

It is normal for the container runtime to use CUDA 11.8 while the host NVIDIA driver reports support for a newer CUDA version. The newer driver is backward compatible with the CUDA runtime packaged in the container.

## Create Working Directories

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

Create directories for the processing stages:

```bash
mkdir -p "$WORK/input"
mkdir -p "$WORK/mineru"
mkdir -p "$WORK/table_crops/images"
mkdir -p "$WORK/paddleocr"
```

Verify the layout:

```bash
find "$WORK" -maxdepth 2 -type d
```

Expected structure:

```text
work/
├── input/
├── mineru/
├── table_crops/
│   └── images/
└── paddleocr/
```

## Intended Data Flow

```text
PDF
 │
 ▼
MinerU
 │
 ├── document/layout analysis
 ├── table detection
 ├── table bounding boxes
 ├── table crops
 └── structured MinerU output
 │
 ▼
table_crops/
 │
 ▼
PaddleOCR-VL
 │
 ├── table content recognition
 ├── row/column structure recognition
 └── structured table reconstruction
 │
 ▼
paddleocr/
```

## Use Separate Conda Environments

MinerU and PaddleOCR should be kept in separate Conda environments because both install substantial machine-learning dependency stacks.

Do not install all dependencies into a general `tabulus` environment.

## MinerU Environment

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

Verify the Python executable:

```bash
python --version
which python
```

Expected path:

```text
~/miniconda3/envs/tabulus-mineru/bin/python
```

## MinerU Installation

Install MinerU in the MinerU environment:

```bash
python -m pip install --upgrade pip
python -m pip install "mineru[all]==3.4.5"
```

Then verify:

```bash
mineru --version
```

At the time this page was prepared, environment creation and GPU compatibility tests had been verified. Continue validating the MinerU installation and inference steps before marking the full pipeline as fully tested.

## PaddleOCR Environment

Create PaddleOCR in a separate environment. The exact installation command should be validated on the GPU server before this page is marked complete.

Target shape:

```bash
conda create -n tabulus-paddleocr python=3.12 -y
conda activate tabulus-paddleocr
python -m pip install --upgrade pip
```

The PaddleOCR dependency stack should be installed and tested independently from MinerU.

## Notes

- Docker is not required for this workflow.
- On HPC systems where Docker is unavailable, Apptainer can execute OCI/Docker images directly.
- The web UI can be omitted.
- Intermediate outputs should be stored under `work/` and inspected directly.
- Use one GPU during initial testing with `CUDA_VISIBLE_DEVICES=0`.
