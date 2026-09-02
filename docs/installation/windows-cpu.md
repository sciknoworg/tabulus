# Windows CPU Setup

This page records the Windows CPU-only MinerU profiling setup that has been validated for Tabulus.

Use this path when you want to run `tabulus profile` on Windows without Docker, Conda, or an NVIDIA GPU. The CPU-compatible MinerU backend is `pipeline`.

The command examples on this page use Windows Command Prompt (`cmd.exe`) unless PowerShell is shown explicitly.

## Validated Environment

The validated setup was:

- Windows 11
- Python 3.12.10
- standard Python virtual environment
- Tabulus installed from the local repository
- MinerU 3.4.5 with the `pipeline` extra
- PyTorch 2.10.0+cpu
- CUDA unavailable, as expected for CPU-only testing

Expected verification state:

```text
Python 3.12.10
MinerU 3.4.5
PyTorch 2.10.0+cpu
CUDA available: False
```

## Clean Reinstall / Start From Scratch

This section is optional. A first-time installation does not need it.

If you are resetting a local development checkout, run these commands from Windows Command Prompt in the repository root:

```bat
deactivate
rmdir /s /q .venv
rmdir /s /q .pytest_cache
rmdir /s /q src\tabulus.egg-info
for /d /r %d in (__pycache__) do @if exist "%d" rmdir /s /q "%d"
```

`deactivate` is only needed if a virtual environment is active. "Directory not found" messages for cleanup targets are harmless.

These commands remove local Python and development artifacts only. They do not delete source code or previously generated `tabulus-output` profiling results.

## Create The Environment

From the repository root, create the virtual environment explicitly with Python 3.12:

```bat
py -3.12 -m venv .venv
```

If the `py` launcher is ambiguous or points to the wrong interpreter, use the absolute Python 3.12 executable instead.

Command Prompt:

```bat
"C:\Path\To\Python312\python.exe" -m venv .venv
```

PowerShell:

```powershell
& "C:\Path\To\Python312\python.exe" -m venv .venv
```

Activate the environment:

```bat
.venv\Scripts\activate
```

Verify Python:

```bat
python --version
```

## Install Tabulus

Upgrade pip:

```bat
python -m pip install --upgrade pip
```

For normal library use, install Tabulus from the local checkout:

```bat
python -m pip install -e .
```

For development and testing, install the development extra instead:

```bat
python -m pip install -e ".[dev]"
```

You do not need to run `python -m pip install -e .` first if you use `python -m pip install -e ".[dev]"`.

Verify the CLI:

```bat
tabulus --version
```

## Install CPU-Only PyTorch

Install the validated CPU-only PyTorch pins:

```bat
python -m pip install "torch==2.10.0+cpu" "torchvision==0.25.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
```

## Install MinerU Pipeline

Install MinerU 3.4.5 with the CPU-compatible pipeline extra while retaining the CPU PyTorch pins:

```bat
python -m pip install six "mineru[pipeline]==3.4.5" "torch==2.10.0+cpu" "torchvision==0.25.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
```

`six` is included here as a compatibility workaround for the tested MinerU 3.4.5 Windows CPU setup. MinerU 3.4.5's bundled OCR implementation imports `six`, but that package is not declared in its `pipeline` dependency set. This is not a Tabulus dependency, and it should not be generalized to later MinerU releases unless those versions are verified.

For the MinerU options and output artifacts used by Tabulus, see {doc}`../external-tools/mineru`.

## Verify The Installation

Check the versions and CUDA state:

```bat
python --version
mineru --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The expected state is:

```text
Python 3.12.10
MinerU 3.4.5
2.10.0+cpu
False
```

## Complete Command Prompt Sequence

From a clean checkout, the complete validated Windows CPU setup and profiling sequence is:

```bat
"C:\Path\To\Python312\python.exe" -m venv .venv
.venv\Scripts\activate

python --version
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
tabulus --version

python -m pip install "torch==2.10.0+cpu" "torchvision==0.25.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu

python -m pip install six "mineru[pipeline]==3.4.5" "torch==2.10.0+cpu" "torchvision==0.25.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu

python --version
mineru --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

python -m pytest -v

tabulus profile --pdf "C:\path\to\paper.pdf" --backend pipeline
```

## Run CPU Profiling

Run MinerU through the Tabulus CLI with the CPU-compatible backend:

```bat
tabulus profile --pdf "C:\path\to\paper.pdf" --backend pipeline
```

`tabulus profile` writes MinerU stdout, stderr, and Tabulus run metadata logs into the profiling output tree.

After a successful MinerU run, `tabulus profile` also exports canonical MinerU table crops into a normalized Tabulus handoff by default.

If `--out` is omitted, Tabulus writes to:

```text
<PDF directory>\tabulus-output\<profiler>\<backend>\
```

For MinerU pipeline profiling, that means:

```text
<PDF directory>\tabulus-output\mineru\pipeline\
```

`mineru` is the profiler. `pipeline` and `hybrid-engine` are MinerU backends.

The `tabulus-output\mineru\pipeline\` directory is the profiler/backend output root passed to MinerU. MinerU then keeps its own native document/run hierarchy underneath that root:

```text
<PDF directory>\
  tabulus-output\
    mineru\
      pipeline\
        <PDF stem>\
          <MinerU-native run directory>\
```

For the validated MinerU 3.4.5 Windows CPU run with `pipeline` + `auto`, MinerU produced the native run directory `auto`:

```text
<papers-folder>\
  tabulus-output\
    mineru\
      pipeline\
        <document>\
          auto\
            images\
            <document>_content_list.json
            <document>_content_list_v2.json
            <document>_layout.pdf
            <document>_middle.json
            <document>_model.json
            mineru_stdout.log
            mineru_stderr.log
            tabulus_run.txt
```

The levels are:

- `mineru`: profiler
- `pipeline`: backend
- `<document>`: document directory
- `auto`: MinerU-native run directory observed for this validated `pipeline` + `auto` run

`auto` is meaningful MinerU output behavior, not a generic Tabulus directory. Native run-directory naming belongs to MinerU and can differ by backend; the validated `hybrid-engine` + `auto` GPU workflow produced `hybrid_auto`.

Do not flatten or rename MinerU-native output files. Tabulus discovers the nested `*_content_list.json` and referenced images from that output tree.

The default normalized table-crop handoff is separate from the native MinerU output:

```text
<PDF directory>\tabulus-output\table-crops\<PDF stem>\
  tables_index.json
  images\
```

Use `--table-crops-out PATH` to override that handoff directory, or `--no-export-table-crops` to skip automatic crop export.

Use `--out` only when you want to override the profiler output root. It is not the final directory for one document. For example:

```bat
tabulus profile --pdf "C:\papers\paper.pdf" --out "D:\results\mineru\pipeline" --backend pipeline
```

causes MinerU to create approximately:

```text
D:\results\mineru\pipeline\paper\auto\
```

If you omit `--backend`, Tabulus opens an interactive backend selector:

```text
1. pipeline       CPU-compatible [default]
2. hybrid-engine  GPU-accelerated
```

Choose `pipeline` for CPU-only Windows runs. `hybrid-engine` requires a suitable CUDA GPU.

If `hybrid-engine` is requested but GPU requirements are not satisfied, Tabulus reports the reason and falls back to `pipeline`. Common fallback reasons include PyTorch not being installed, CUDA not being available, no visible CUDA GPU, insufficient GPU architecture, or insufficient VRAM. When Tabulus generates the output directory automatically, it uses the resolved backend name, so a fallback run writes under `tabulus-output\mineru\pipeline\`.

## Validated Windows Run

MinerU 3.4.5 `pipeline` completed a real 53-page PDF profiling run on Windows CPU with:

```text
Python 3.12.10
PyTorch 2.10.0+cpu
CUDA available: False
MinerU 3.4.5
```

The Windows test suite was run with:

`pytest` is provided by the Tabulus `dev` extra. If Tabulus was installed only with `python -m pip install -e .`, install `python -m pip install -e ".[dev]"` before running the test suite.

```bat
python -m pytest -v
```

and passed in this environment:

```text
21 passed
```

with pytest 9.1.1.

## Inspect Tables After Profiling

After MinerU writes its output directory, the library can discover table regions:

```bat
python -c "from pathlib import Path; from tabulus.mineru import discover_tables; tables, refs = discover_tables(Path('C:/path/to/papers/tabulus-output/mineru/pipeline/<document>/auto')); print(len(tables)); print(refs)"
```

`tabulus profile` prepares the table-crop handoff automatically by default. To regenerate that handoff from an existing MinerU run without rerunning MinerU:

```bat
tabulus export-table-crops --mineru-root "C:\path\to\papers\tabulus-output\mineru\pipeline\<document>\auto" --out "work\table_crops"
```

This writes:

```text
work\table_crops\
  tables_index.json
  images\
```

Table reconstruction adapters operate on canonical MinerU table crops. If the PaddleOCR dependencies are installed in the active environment, the batch table-reconstruction CLI can process the crop handoff with PaddleOCR-VL:

```bat
tabulus reconstruct-tables --crops "C:\path\to\papers\tabulus-output\table-crops\<document>" --adapter paddleocr-vl --device cpu
```

The command writes adapter outputs under:

```text
<crop-root>\reconstructions\paddleocr-vl\
  native\
  parsed\
  predictions\
  batch_summary.json
```

Reference matching, DOI resolution, continued-table merging, final resolved CSV export, and full end-to-end processing are not yet implemented in the new library.
