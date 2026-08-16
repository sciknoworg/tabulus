# Tabulus

**Scientific PDF Table Extraction Pipeline**

## About Tabulus

Tabulus is a modular pipeline for digitizing scientific PDF papers into structured, reference-aware table data.

The project is being reorganized around standalone processing components rather than around specific OCR or extraction libraries. Each component should be runnable on its own, exchange data through a documented contract, and plug into the end-to-end pipeline once the individual step is stable.

The current validated library module focuses on MinerU-backed PDF profiling. Tabulus can launch MinerU through the `tabulus profile` command, read the resulting structured output, expose typed table regions, and export a normalized table-crop handoff with image paths, page numbers, bounding boxes, captions, footnotes, and MinerU `table_body` values.

## How To Use This Documentation

Start with the installation page that matches your machine. Windows and CPU-only users can use MinerU's `pipeline` backend. GPU-server users can use MinerU's `hybrid-engine` backend when a suitable CUDA GPU is visible. Library contributors can use the core Python setup for GPU-independent unit tests.

For the current implementation, the most important distinction is:

- **Already validated:** cross-platform MinerU execution through `tabulus profile`, a real 53-page Windows CPU profiling run with MinerU `pipeline`, GPU-server profiling with MinerU `hybrid-engine`, typed table-region discovery with `tabulus.mineru`, and `tables_index.json` table-crop export.
- **Next stages:** PaddleOCR-VL execution, reference processing, run reporting, and full end-to-end commands.

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

Prepare the Linux GPU-server workflow, separate Conda environments, and MinerU `hybrid-engine`.
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

## Current Status

The current library can:

- launch MinerU through `tabulus profile`
- write profiling output to `<PDF directory>/tabulus-output/<PDF stem>/profiling/<profiler>/<backend>/` when `--out` is omitted
- distinguish the profiler (`mineru`) from MinerU backends (`pipeline` and `hybrid-engine`)
- locate MinerU `*_content_list.json` files
- parse document elements
- select table regions
- resolve MinerU-generated table images
- convert zero-based MinerU page indices into document page numbers
- retain provenance and expose typed `TableRegion` objects
- export table images plus `tables_index.json` through `tabulus export-table-crops`

The current library does not yet run PaddleOCR-VL, run bibliography/reference matching, produce run reports, or write final CSV outputs.

## Documentation Map

The sidebar contains the full documentation. The main sections are:

- **Installation And Setup:** Windows CPU, GPU server, and Python library setup.
- **Tutorial:** the intended modular workflow, one processing step at a time.
- **Components:** adapter boundaries and responsibilities.
- **Workflows:** headless local/GPU execution shapes.
- **Data Contracts:** the file formats exchanged between modules.
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
tutorial/09-table-normalization
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
:caption: Data Contracts

data-contracts/run-directory
data-contracts/mineru-output-files
data-contracts/pdf-profile-json
data-contracts/tables-index-json
data-contracts/ocr-tables-json
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
