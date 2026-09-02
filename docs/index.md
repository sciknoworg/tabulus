# Tabulus

**Scientific PDF Table Extraction Pipeline**

## About Tabulus

Tabulus is a modular pipeline for digitizing scientific PDF papers into structured, reference-aware table data.

The project is being reorganized around standalone processing components rather than around specific OCR or extraction libraries. Each component should be runnable on its own, exchange data through a documented contract, and plug into the end-to-end pipeline once the individual step is stable.

The current validated library module focuses on MinerU-backed PDF profiling and batch table reconstruction from canonical MinerU crops. Tabulus can launch MinerU through the `tabulus profile` command, read the resulting structured output, expose typed table regions, and export a normalized table-crop handoff with image paths, page numbers, bounding boxes, captions, footnotes, and MinerU `table_body` values. The table reconstruction layer provides an extensible adapter contract with registered PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, Granite Vision 4.1 4B, TRivia-3B, GLM-OCR, Dolphin-v2, DeepSeek-OCR-2, Nanonets-OCR-s, MonkeyOCRv2-B-Parsing, NVIDIA Nemotron Parse v1.2, HunyuanOCR-1.5, and dots.mocr adapters, plus `tabulus reconstruct-tables` for adapter-neutral batch reconstruction.

The target full pipeline keeps several output layers distinct: external-tool native artifacts, parsed table-reconstruction artifacts used for evaluation, prediction CSV files, bibliography/reference-resolution artifacts, and final resolved CSV files. The current implementation has validated MinerU, PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, Granite Vision 4.1 4B, TRivia-3B, GLM-OCR, Dolphin-v2, DeepSeek-OCR-2, Nanonets-OCR-s, MonkeyOCRv2-B-Parsing, NVIDIA Nemotron Parse v1.2, HunyuanOCR-1.5, and dots.mocr integration pieces; the later bibliography, matching, DOI, and resolved-CSV stages remain planned for the rebuilt library.

## How To Use This Documentation

Start with the installation page that matches your machine. Windows and CPU-only users can use MinerU's `pipeline` backend. GPU-server users can use MinerU's `hybrid-engine` backend when a suitable CUDA GPU is visible. Library contributors can use the core Python setup for GPU-independent unit tests.

For the current implementation, the most important distinction is:

- **Already validated:** cross-platform MinerU execution through `tabulus profile`, typed table-region discovery with `tabulus.mineru`, automatic export of canonical table crops, PaddleOCR-VL CPU/GPU inference on MinerU table crops, Chandra OCR 2 GPU reconstruction, NuExtract3 GPU reconstruction, Tesseract + Table Transformer reconstruction, RapidOCR + Docling TableFormer reconstruction, Granite Vision 4.1 4B reconstruction, TRivia-3B reconstruction, GLM-OCR reconstruction, Dolphin-v2 reconstruction, DeepSeek-OCR-2 reconstruction, Nanonets-OCR-s reconstruction, MonkeyOCRv2-B-Parsing reconstruction, NVIDIA Nemotron Parse v1.2 reconstruction, HunyuanOCR-1.5 reconstruction, dots.mocr reconstruction, `tabulus reconstruct-tables`, batch reconstruction dispatch, legacy-compatible HTML/Markdown table parsing, deterministic OTSL-to-HTML normalization, and `tabulus classify-reference-tables`.
- **Planned stages:** bibliography extraction, reference matching, DOI resolution, resolved CSV export, run reporting, and full end-to-end commands.

## Where To Start

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} CPU / Windows Setup
:link: installation/windows-cpu
:link-type: doc

Use Python 3.12, a standard venv, CPU-only PyTorch, and MinerU `pipeline`.
:::

:::{grid-item-card} GPU Server Setup
:link: installation/gpu-server
:link-type: doc

Request Slurm GPU resources, install the `tabulus-mineru` environment, and run MinerU `hybrid-engine`.
:::

:::{grid-item-card} Install The Python Library
:link: installation/python-library
:link-type: doc

Install Tabulus for core library development and run GPU-independent unit tests.
:::

:::{grid-item-card} Run MinerU On GPU
:link: workflows/mineru-gpu-execution
:link-type: doc

Use the tested MinerU 3.4.5 command sequence and validate the output with `discover_tables`.
:::

:::{grid-item-card} Follow The Core Tutorial
:link: tutorial/00-overview
:link-type: doc

Read the modular pipeline overview and the ordered processing steps.
:::

::::

## Documentation Map

The sidebar contains the full documentation. The main sections are:

- **Installation And Setup:** Windows CPU, GPU server, and Python library setup.
- **Tutorial:** the intended modular workflow, one processing step at a time.
- **Components:** adapter boundaries and responsibilities.
- **Workflows:** headless local/GPU execution shapes.
- **External Tools:** third-party tools as used by Tabulus.
- **Data Contracts:** the file formats and artifact layers exchanged between modules.
- **Evaluation:** how to compare extraction and matching quality.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Installation And Setup

installation/windows-cpu
installation/gpu-server
installation/python-library
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Tutorial

tutorial/00-overview
tutorial/01-pdf-profiling
tutorial/08-table-ocr
tutorial/10-reference-table-classification
tutorial/11-bibliography-extraction
tutorial/12-reference-matching
tutorial/13-doi-resolution
tutorial/14-csv-export
tutorial/15-run-report
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Components

modules/overview
modules/pdf-profilers
modules/pdf-readers
modules/page-renderers
modules/layout-detectors
modules/table-detectors
modules/table-ocr-adapters
modules/bibliography-extractors
modules/reference-matchers
modules/doi-resolvers
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Workflows

workflows/single-pdf-local
workflows/gpu-server-run
workflows/mineru-gpu-execution
workflows/end-to-end-run
workflows/debugging-failed-step
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: External Tools

external-tools/mineru
external-tools/paddleocr-vl
external-tools/chandra
external-tools/nuextract3
external-tools/tesseract-tatr
external-tools/docling
external-tools/granite-vision
external-tools/trivia
external-tools/glm-ocr
external-tools/dolphin-v2
external-tools/deepseek-ocr-2
external-tools/nanonets-ocr-s
external-tools/monkeyocrv2-b-parsing
external-tools/nemotron-parse-v1-2
external-tools/hunyuanocr-1-5
external-tools/dots-mocr
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Data Contracts

data-contracts/run-directory
data-contracts/mineru-output-files
data-contracts/pdf-profile-json
data-contracts/tables-index-json
data-contracts/ocr-tables-json
data-contracts/table-prediction-csv
data-contracts/bibliography-json
data-contracts/reference-matches-json
data-contracts/resolved-csv
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Evaluation

evaluation/overview
evaluation/table-extraction-quality
evaluation/bibliography-quality
evaluation/reference-matching-quality
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project Notes

project-notes/current-state
project-notes/branding
project-notes/containerization-later
```
