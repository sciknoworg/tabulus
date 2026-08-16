# Windows CPU Setup

This page records the Windows CPU-only MinerU profiling setup that has been validated for Tabulus.

Use this path when you want to run `tabulus profile` on Windows without Docker, Conda, or an NVIDIA GPU. The CPU-compatible MinerU backend is `pipeline`.

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

## Windows Case-Sensitivity Fix

The repository previously contained both `src/Tabulus/` and `src/tabulus/`. On Windows, case-insensitive filesystems can collapse those names into the same physical path, which prevented normal imports of the new `tabulus` package.

The legacy implementation has been renamed to:

```text
src/legacy_tabulus/
```

Do not reintroduce a top-level package or directory whose name differs from `src/tabulus/` only by case.

## Create The Environment

From the repository root, create the virtual environment explicitly with Python 3.12:

```powershell
py -3.12 -m venv .venv
```

If the `py` launcher is ambiguous or points to the wrong interpreter, use the absolute Python 3.12 executable instead:

```powershell
& "C:\Path\To\Python312\python.exe" -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\activate
```

Verify Python:

```powershell
python --version
```

## Install Tabulus

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install Tabulus from the local checkout:

```powershell
python -m pip install -e .
```

If you also intend to run the test suite, install the development extra instead:

```powershell
python -m pip install -e ".[dev]"
```

Verify the CLI:

```powershell
tabulus --version
```

## Install CPU-Only PyTorch

Install the validated CPU-only PyTorch pins:

```powershell
python -m pip install "torch==2.10.0+cpu" "torchvision==0.25.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
```

## Install MinerU Pipeline

Install MinerU 3.4.5 with the CPU-compatible pipeline extra while retaining the CPU PyTorch pins:

```powershell
python -m pip install six "mineru[pipeline]==3.4.5" "torch==2.10.0+cpu" "torchvision==0.25.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
```

`six` is included here as a compatibility workaround for the tested MinerU 3.4.5 Windows CPU setup. MinerU 3.4.5's bundled OCR implementation imports `six`, but that package is not declared in its `pipeline` dependency set. This is not a Tabulus dependency, and it should not be generalized to later MinerU releases unless those versions are verified.

## Verify The Installation

Check the versions and CUDA state:

```powershell
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

## Run CPU Profiling

Run MinerU through the Tabulus CLI with the CPU-compatible backend:

```powershell
tabulus profile --pdf "C:\path\to\paper.pdf" --backend pipeline
```

`tabulus profile` writes MinerU stdout, stderr, and Tabulus run metadata logs into the profiling output directory.

If `--out` is omitted, Tabulus writes to:

```text
<PDF directory>\tabulus-output\<PDF stem>\profiling\<profiler>\<backend>\
```

For MinerU pipeline profiling, that means:

```text
<PDF directory>\tabulus-output\<PDF stem>\profiling\mineru\pipeline\
```

`mineru` is the profiler. `pipeline` and `hybrid-engine` are MinerU backends.

MinerU keeps its own native output hierarchy underneath that profiling directory. For example, a PDF named `Puurunen - February 2005.pdf` can produce:

```text
Puurunen - February 2005\
  tabulus-output\
    Puurunen - February 2005\
      profiling\
        mineru\
          pipeline\
            Puurunen - February 2005\
              auto\
                Puurunen - February 2005_content_list.json
                images\
```

Do not flatten or rename MinerU-native output files. Tabulus discovers the nested `*_content_list.json` and referenced images from that output tree.

Use `--out` only when you want to override the default output location:

```powershell
tabulus profile --pdf "C:\path\to\paper.pdf" --out "C:\path\to\custom-output" --backend pipeline
```

If you omit `--backend`, Tabulus opens an interactive backend selector:

```text
1. pipeline       CPU-compatible [default]
2. hybrid-engine  GPU-accelerated
```

Choose `pipeline` for CPU-only Windows runs. `hybrid-engine` requires a suitable CUDA GPU.

If `hybrid-engine` is requested but GPU requirements are not satisfied, Tabulus reports the reason and falls back to `pipeline`. Common fallback reasons include PyTorch not being installed, CUDA not being available, no visible CUDA GPU, insufficient GPU architecture, or insufficient VRAM. When Tabulus generates the output directory automatically, it uses the resolved backend name, so a fallback run writes under `profiling\mineru\pipeline\`.

## Validated Windows Run

MinerU 3.4.5 `pipeline` completed a real 53-page PDF profiling run on Windows CPU with:

```text
Python 3.12.10
PyTorch 2.10.0+cpu
CUDA available: False
MinerU 3.4.5
```

The Windows test suite also passed in this environment:

```text
21 passed
```

with pytest 9.1.1.

## Inspect Tables After Profiling

After MinerU writes its output directory, the library can discover table regions:

```powershell
python -c "from pathlib import Path; from tabulus.mineru import discover_tables; tables, refs = discover_tables(Path('C:/path/to/profiling/mineru/pipeline')); print(len(tables)); print(refs)"
```

To prepare the table-crop handoff for later OCR work:

```powershell
tabulus export-table-crops --mineru-root "<output-dir>" --out "work\table_crops"
```

This writes:

```text
work\table_crops\
  tables_index.json
  images\
```

PaddleOCR-VL, reference matching, DOI resolution, and full end-to-end processing are not yet implemented in the new library.
