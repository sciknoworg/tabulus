# Tabulus

**Scientific PDF Table Extraction Pipeline**

## About Tabulus

Tabulus is a modular pipeline for digitizing scientific PDF papers into
structured, reference-aware table data.

The rebuilt library is organized around standalone commands and explicit
filesystem contracts. The current runnable workflow covers PDF profiling,
canonical table-crop export, table reconstruction, and reference-table
classification. Later bibliography extraction, reference matching, DOI
resolution, resolved CSV export, run reports, and complete `tabulus run`
orchestration remain planned.

Start with the commands, then use the linked pages for setup and adapter
details. For one PDF:

```bash
tabulus profile --pdf /path/to/paper.pdf --backend pipeline

tabulus reconstruct-tables \
  --crops /path/to/tabulus-output/table-crops/<paper> \
  --adapter <adapter> \
  --device gpu:0

tabulus classify-reference-tables \
  --reconstruction /path/to/tabulus-output/table-crops/<paper>/reconstructions/<adapter>
```

For several PDFs in one folder:

```bash
tabulus profile \
  --folder /path/to/papers \
  --backend hybrid-engine \
  --method auto \
  --effort high

tabulus reconstruct-tables \
  --crops-folder /path/to/papers/tabulus-output/table-crops \
  --adapter <adapter> \
  --device gpu:0

tabulus classify-reference-tables \
  --crops-folder /path/to/papers/tabulus-output/table-crops \
  --adapter <adapter>
```

## How To Use This Documentation

Start with the installation page that matches your machine. Windows and
CPU-only users can use MinerU's `pipeline` backend. GPU-server users can use
MinerU's `hybrid-engine` backend and adapter-specific reconstruction
environments when a suitable CUDA GPU is visible. Library contributors can use
the core Python setup for GPU-independent unit tests.

For the current adapter list and command examples, see
{doc}`tutorial/08-table-ocr`.

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

Request GPU resources, install stage-specific environments, and run MinerU or
Stage 2 reconstruction adapters.
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

Read the current runnable stages and artifact flow.
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
external-tools/internvl3-5-8b
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
