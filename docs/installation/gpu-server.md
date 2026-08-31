# GPU Server Installation

This page documents the supported GPU installation and validation workflow for Tabulus. MinerU profiling uses the `tabulus-mineru` Conda environment and MinerU's `hybrid-engine` backend. PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, Granite Vision 4.1 4B, TRivia-3B, GLM-OCR, Dolphin-v2, and DeepSeek-OCR-2 table reconstruction are validated separately in adapter-specific environments so their heavyweight dependency stacks do not destabilize each other.

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
     |
     v
tabulus reconstruct-tables --crops "$PAPERS/tabulus-output/table-crops/..." --adapter <adapter>
     |
     v
prediction CSV files and batch_summary.json
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
- separate Conda environments for MinerU, PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, Granite Vision 4.1 4B, TRivia-3B, GLM-OCR, Dolphin-v2, and DeepSeek-OCR-2

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

The automated test suite uses mocks for heavyweight reconstruction dependencies and does not replace Linux GPU integration validation for the reconstruction adapters.

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

After successful profiling, Tabulus also exports canonical MinerU table crops automatically by default to:

```text
$PAPERS/
└── tabulus-output/
    └── table-crops/
        └── <document>/
            ├── tables_index.json
            └── images/
```

Use `--table-crops-out PATH` to override the normalized handoff directory, or `--no-export-table-crops` to skip automatic crop export.

The validated Linux GPU run regenerated the Puurunen PDF outputs with MinerU `hybrid-engine` and automatically exported 23 canonical table crops.

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

## 11. Run Batch Table Reconstruction

Run table reconstruction in the environment that matches the selected adapter.
The separation is intentional:

```text
tabulus-mineru
  Tabulus + MinerU + PyTorch

tabulus-paddleocr-gpu
  Tabulus + PaddleOCR + PaddlePaddle

tabulus-chandra-gpu
  Tabulus + Chandra OCR + PyTorch/Transformers

tabulus-nuextract3-gpu
  Tabulus + NuExtract3 + PyTorch/Transformers/Accelerate

tabulus-tesseract-tatr-gpu
  Tabulus + Tesseract + Table Transformer + PyTorch/Transformers

dedicated RapidOCR + Docling TableFormer environment
  Tabulus + RapidOCR + ONNX Runtime + Docling TableFormer

tabulus-granite-vision
  Tabulus + Granite Vision 4.1 4B + Docling + Transformers

tabulus-trivia-gpu
  Tabulus + TRivia-3B + PyTorch/Transformers/Accelerate

tabulus-glm-ocr-gpu
  Tabulus + GLM-OCR + PyTorch/Transformers/Accelerate

tabulus-dolphin-v2-gpu
  Tabulus + Dolphin-v2 + PyTorch/Transformers/Accelerate/qwen-vl-utils

tabulus-deepseek-ocr-2-gpu
  Tabulus + DeepSeek-OCR-2 + PyTorch/Transformers/FlashAttention
```

These environments can install Tabulus from the same repository checkout in editable mode. They are pipeline-stage environments, not separate versions of the Tabulus source code.

### PaddleOCR-VL GPU Environment

The validated PaddleOCR-VL GPU stack was:

- Python 3.12
- PaddlePaddle-GPU 3.2.1
- PaddleOCR 3.7.0
- PaddleOCR-VL 1.6
- NVIDIA L40S
- `device="gpu:0"`
- `engine="paddle"`

PaddleOCR-VL is applied only to the canonical MinerU table crops:

```text
$PAPERS/
└── tabulus-output/
    └── table-crops/
        └── <document>/
            ├── tables_index.json
            └── images/
                └── page_006_table_001.jpg
```

The validated adapter configuration disables layout detection and enables the table prompt:

```python
PaddleOCRVL(
    pipeline_version="v1.6",
    device="gpu:0",
    engine="paddle",
    use_layout_detection=False,
)

pipeline.predict(
    str(image_path),
    use_layout_detection=False,
    prompt_label="table",
)
```

For `page_006_table_001.jpg`, the GPU run succeeded and produced one parsed HTML table with 58 rows x 6 columns.

Repeatability observations using the same loaded adapter and the same crop:

```text
first cached-model pass: 44.58 s
warm second pass:       25.24 s
parsed table shape:     58 x 6 both times
parsed cell differences: 0
```

The first-ever GPU run took 91.97 s because it also included model download and setup. Treat these timings as validation observations, not formal benchmarks.

The Windows CPU and Linux GPU crops were not byte-identical. The observed crop dimensions were 1431 x 1923 on Windows CPU and 1432 x 1923 on Linux GPU, so do not draw strong CPU-vs-GPU accuracy conclusions from output differences alone.

The implemented batch CLI reuses one adapter instance across every crop in the handoff:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus reconstruct-tables \
  --crops "$PAPERS/tabulus-output/table-crops/<document>" \
  --adapter paddleocr-vl \
  --device gpu:0
```

If `--out` is omitted, the command writes:

```text
$PAPERS/
└── tabulus-output/
    └── table-crops/
        └── <document>/
            └── reconstructions/
                └── paddleocr-vl/
                    ├── native/
                    ├── parsed/
                    ├── predictions/
                    └── batch_summary.json
```

`native/` preserves the full adapter result and provenance. `parsed/` preserves the rectangular parsed table representation. `predictions/` contains pre-reference-resolution CSV files. `batch_summary.json` records counts, per-table status, runtime, artifact paths, and errors.

This command reconstructs physical MinerU crops independently. It does not merge continued tables, classify reference tables, extract bibliographies, match references, resolve DOI values, or write final resolved CSV files.

### Chandra OCR 2 GPU Environment

Chandra OCR 2 runs in its own environment because it uses a PyTorch and
Transformers stack rather than PaddlePaddle. The validated Chandra environment
was:

- environment name: `tabulus-chandra-gpu`
- Python 3.12.13
- `chandra-ocr[hf]` 0.2.0
- PyTorch 2.13.0+cu130
- Transformers 5.15.1
- NVIDIA L40S
- CUDA available
- Tabulus installed editable with development dependencies during validation

Create and activate the environment:

```bash
conda create -n tabulus-chandra-gpu python=3.12.13 -y
conda activate tabulus-chandra-gpu
```

Install Tabulus and Chandra:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install "chandra-ocr[hf]==0.2.0"
```

Verify the runtime from inside the environment:

```bash
python --version
python -m pip show chandra-ocr torch transformers
python - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
print("Transformers:", transformers.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

The validated run resolved `--device gpu:0` to PyTorch `cuda:0`.

Chandra consumes the canonical MinerU crop handoff. It should not be run
against the original PDFs for this reconstruction comparison:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus reconstruct-tables \
  --crops-folder "$PAPERS/tabulus-output/table-crops" \
  --adapter chandra \
  --device gpu:0
```

The batch layer loads one Chandra model instance and reuses it across the
complete batch. For each paper crop root, output is written beside other
adapters under:

```text
<crop-root>/
  reconstructions/
    chandra/
      native/
      parsed/
      predictions/
      batch_summary.json
```

`native/` preserves the generated raw HTML plus Chandra metadata such as token
count and generation error status. `parsed/`, `predictions/`, and
`batch_summary.json` follow the same Tabulus reconstruction output contract as
PaddleOCR-VL.

Hugging Face authentication is optional for public model access, but
authenticated access avoids unauthenticated Hub warnings and rate limits. Set
the token as an environment variable when needed:

```bash
export HF_TOKEN="<your-hugging-face-token>"
```

Never hard-code an actual token in documentation, scripts, or committed files.

Current Transformers releases may emit warnings mentioning `min_frames`,
`max_frames`, or `processor_kwargs`. These warnings were non-fatal in the
validated Chandra runs and should not be treated as Tabulus errors by
themselves.

### NuExtract3 GPU Environment

NuExtract3 runs in its own GPU-capable environment because it uses a PyTorch,
Transformers, Accelerate, and Pillow stack. It consumes the same canonical
MinerU crop handoff as the other reconstruction adapters and should not be run
against the original PDFs for this reconstruction comparison.

The implemented NuExtract3 adapter is GPU-only in the validated Tabulus
configuration and is registered as `nuextract3`.

Run reconstruction with:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus reconstruct-tables \
  --crops-folder "$PAPERS/tabulus-output/table-crops" \
  --adapter nuextract3 \
  --device gpu:0
```

For each paper crop root, output is written beside other adapters under:

```text
<crop-root>/
  reconstructions/
    nuextract3/
      native/
      parsed/
      predictions/
      batch_summary.json
```

NuExtract3 is invoked through Hugging Face Transformers in-process with
`mode="markdown"`, `enable_thinking=False`, and deterministic generation.
Tabulus does not require a vLLM HTTP service for this adapter path.

For the NuExtract3 settings and output artifacts used by Tabulus, see
{doc}`../external-tools/nuextract3`.

### Tesseract + Table Transformer GPU Environment

Tesseract + Table Transformer runs in its own environment because it combines
the external Tesseract executable with a PyTorch/Transformers Table Transformer
model stack. It consumes the same canonical MinerU crop handoff as the other
reconstruction adapters and should not be run against the original PDFs for
this reconstruction comparison.

The implemented adapter is registered as `tesseract-tatr`.

Run reconstruction with:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus reconstruct-tables \
  --crops-folder "$PAPERS/tabulus-output/table-crops" \
  --adapter tesseract-tatr \
  --device gpu:0
```

For each paper crop root, output is written beside other adapters under:

```text
<crop-root>/
  reconstructions/
    tesseract-tatr/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Tesseract performs OCR/text recognition and word bounding-box extraction.
Microsoft Table Transformer performs table-structure recognition using the
`microsoft/table-transformer-structure-recognition-v1.1-all` model. Tabulus
then fuses tokens and structure deterministically and passes the generated HTML
through the shared parser.

For the Tesseract + Table Transformer settings and output artifacts used by
Tabulus, see {doc}`../external-tools/tesseract-tatr`.

### RapidOCR + Docling TableFormer Environment

RapidOCR + Docling TableFormer consumes the same canonical MinerU crop handoff
as the other reconstruction adapters. RapidOCR with ONNX Runtime performs OCR
and word-bounding-box extraction on the crop using the CPU. Docling's
TableFormer V1 then performs table-structure recognition on the complete crop
using the requested CPU or GPU device.

Run reconstruction with:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus reconstruct-tables \
  --crops-folder "$PAPERS/tabulus-output/table-crops" \
  --adapter rapidocr-tableformer \
  --device gpu:0
```

Tabulus uses Docling's bare-crop TableFormer path. It does not run Docling PDF
or page-layout detection, redetect or recrop tables, semantically correct cell
content, or merge continued tables. Raw OTSL and the final Docling table
structure are preserved as native adapter evidence before the common Tabulus
parser produces the shared representation and prediction CSVs.

For the RapidOCR + Docling TableFormer integration details and limitations, see
{doc}`../external-tools/docling`.

### Granite Vision 4.1 4B GPU Environment

Granite Vision is a GPU-only reconstruction adapter in the validated Tabulus
configuration. It consumes the same canonical MinerU crop handoff as the other
adapters. The model receives the complete crop directly and generates OTSL
containing both table structure and cell text; there is no separate OCR engine
or Docling PDF/layout/table-detection step in this adapter.

Create and activate a dedicated environment:

```bash
conda create -n tabulus-granite-vision python=3.12 -y
conda activate tabulus-granite-vision
cd "$TABULUS_ROOT"
```

Install the validated Tabulus and model dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pip install "docling[vlm]==2.123.1"
python -m pip install --upgrade --force-reinstall \
  "transformers==4.57.3"
```

Transformers 4.57.3 is pinned here because the validated Granite integration
required it. A newer Transformers 5.x resolution in the initial environment
was not compatible with this model's generation path; this is a validated
configuration constraint, not a universal statement about future releases.

Before running the adapter, check CUDA visibility from this environment:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Run reconstruction against the canonical MinerU crops:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus reconstruct-tables \
  --crops-folder "$PAPERS/tabulus-output/table-crops" \
  --adapter granite-vision-table \
  --device gpu:0
```

The adapter preserves the Granite model revision, raw generated output, OTSL
sequence, structured cells and dimensions, image dimensions, and device and
version metadata under the standard `native/` layer before shared parsing.
For the full integration details and output boundaries, see
{doc}`../external-tools/granite-vision`.

### TRivia-3B GPU Environment

TRivia-3B is a GPU-only reconstruction adapter in the current Tabulus
configuration. It consumes the same canonical MinerU crop handoff as the other
adapters and sends each crop directly to `opendatalab/TRivia-3B`; there is no
Docker service, vLLM service, `qwen-vl-utils` requirement, original-PDF
redetection, or candidate-specific recropping.

Create and activate a dedicated environment:

```bash
conda create -n tabulus-trivia-gpu python=3.12 -y
conda activate tabulus-trivia-gpu
cd "$TABULUS_ROOT"
```

Install Tabulus and the validated TRivia runtime pieces:

```bash
python -m pip install -e ".[dev]"
python -m pip install "transformers==5.16.1" accelerate pillow
```

Install a CUDA-capable PyTorch build appropriate for the GPU server before
running the adapter, then verify CUDA visibility from inside this environment:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
print("Transformers:", transformers.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Run reconstruction against the canonical MinerU crops:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus reconstruct-tables \
  --crops-folder "$PAPERS/tabulus-output/table-crops" \
  --adapter trivia \
  --device gpu:0
```

The adapter preserves TRivia model/revision metadata, generation settings,
token counts, raw OTSL, image dimensions, and Tabulus OTSL normalization
provenance under the standard `native/` layer before shared parsing. For the
full integration details and output boundaries, see
{doc}`../external-tools/trivia`.

### GLM-OCR GPU Environment

GLM-OCR is a GPU-only reconstruction adapter in the current Tabulus
configuration. It consumes the same canonical MinerU crop handoff as the other
adapters and sends each crop directly to `zai-org/GLM-OCR`; Tabulus does not
invoke the GLM-OCR SDK document pipeline, PP-DocLayout-V3, vLLM, SGLang,
Docker, a hosted API, original-PDF redetection, or candidate-specific
recropping.

Create and activate a dedicated environment:

```bash
conda create -n tabulus-glm-ocr-gpu python=3.12 -y
conda activate tabulus-glm-ocr-gpu
cd "$TABULUS_ROOT"
```

Install Tabulus and the validated GLM-OCR runtime pieces:

```bash
python -m pip install -e ".[dev]"
python -m pip install "transformers==5.16.1" accelerate pillow
```

Install a CUDA-capable PyTorch build appropriate for the GPU server before
running the adapter, then verify CUDA visibility from inside this environment:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
print("Transformers:", transformers.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Run reconstruction against the canonical MinerU crops:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus reconstruct-tables \
  --crops-folder "$PAPERS/tabulus-output/table-crops" \
  --adapter glm-ocr \
  --device gpu:0
```

The adapter preserves GLM-OCR model/revision metadata, raw generated HTML,
clean parser-facing HTML with model special tokens removed, resolved dtype and
device metadata, image dimensions, and generation provenance under the
standard `native/` layer before shared HTML parsing. For the full integration
details and output boundaries, see {doc}`../external-tools/glm-ocr`.

### Dolphin-v2 GPU Environment

Dolphin-v2 is a GPU-only reconstruction adapter in the current Tabulus
configuration. It consumes the same canonical MinerU crop handoff as the other
adapters and sends each crop directly to the `ByteDance/Dolphin-v2` checkpoint.
The checkpoint uses a Qwen2.5-VL backbone, but Tabulus is not substituting a
generic Qwen checkpoint for Dolphin-v2.

Create and activate a dedicated environment:

```bash
conda create -n tabulus-dolphin-v2-gpu python=3.12 -y
conda activate tabulus-dolphin-v2-gpu
cd "$TABULUS_ROOT"
```

Install Tabulus and the validated Dolphin-v2 runtime pieces:

```bash
python -m pip install -e ".[dev]"
python -m pip install \
  "torch==2.6.0" \
  "torchvision==0.21.0" \
  "transformers==4.51.0" \
  "accelerate==1.4.0" \
  "qwen-vl-utils==0.0.14"
```

The adapter requires a CUDA-capable environment in the current Tabulus
configuration. The versions above describe the validated software environment;
do not treat them as claims about all future Dolphin-v2 or Transformers
releases.

Verify CUDA visibility from inside this environment:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
print("Transformers:", transformers.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Run reconstruction against the canonical MinerU crops:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus reconstruct-tables \
  --crops-folder "$PAPERS/tabulus-output/table-crops" \
  --adapter dolphin-v2 \
  --device gpu:0
```

`CUDA_VISIBLE_DEVICES` can remap physical GPU numbering. For example, a
physical GPU selected by the scheduler may appear to the process as `cuda:0`.

The adapter preserves Dolphin-v2 model/revision metadata, backbone and model
class, raw generated HTML, clean parser-facing HTML with model special tokens
removed, deterministic generation settings, source and resized image
dimensions, token counts, and image-preprocessing provenance under the
standard `native/` layer before shared HTML parsing. For the full integration
details and output boundaries, see {doc}`../external-tools/dolphin-v2`.

### DeepSeek-OCR-2 GPU Environment

DeepSeek-OCR-2 is a GPU-only reconstruction adapter in the current Tabulus
configuration. It consumes the same canonical MinerU crop handoff as the other
adapters and sends each crop directly to the `deepseek-ai/DeepSeek-OCR-2`
checkpoint. The adapter uses the model-specific `infer(...)` path with custom
Transformers model code from the pinned Hugging Face model revision.

Create and activate a dedicated environment:

```bash
conda create -n tabulus-deepseek-ocr-2-gpu python=3.12 -y
conda activate tabulus-deepseek-ocr-2-gpu
cd "$TABULUS_ROOT"
```

Install Tabulus and the validated DeepSeek-OCR-2 runtime pieces:

```bash
python -m pip install -e ".[dev]"
python -m pip install \
  "torch==2.6.0" \
  "torchvision==0.21.0" \
  "transformers==4.46.3" \
  "tokenizers==0.20.3" \
  "flash-attn==2.7.3" \
  pillow einops addict easydict
```

The adapter explicitly validates `transformers==4.46.3`,
`tokenizers==0.20.3`, and `flash-attn==2.7.3`. The other packages shown are
part of the validated runtime environment but are not all version-pinned by
the adapter itself.

Verify CUDA visibility from inside this environment:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
print("Transformers:", transformers.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Run reconstruction against the canonical MinerU crops:

```bash
CUDA_VISIBLE_DEVICES=0 tabulus reconstruct-tables \
  --crops-folder "$PAPERS/tabulus-output/table-crops" \
  --adapter deepseek-ocr-2 \
  --device gpu:0
```

The adapter records `input_policy=canonical_mineru_crop`,
`layout_redetection=False`, `recropping=False`, and
`external_recropping=False`. Its `crop_mode=True` setting is DeepSeek's
model-internal dynamic-resolution tiling of the already supplied crop, not
external table redetection or recropping. For the full integration details and
output boundaries, see {doc}`../external-tools/deepseek-ocr-2`.
