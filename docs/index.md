# Tabulus

**Scientific PDF Table Extraction Pipeline**

## About Tabulus

Tabulus is a modular pipeline for digitizing scientific PDF papers into structured, reference-aware table data.

The project is being reorganized around standalone processing components rather than around specific OCR or extraction libraries. Each component should be runnable on its own, exchange data through a documented contract, and plug into the end-to-end pipeline once the individual step is stable.

The current validated library module focuses on typed access to MinerU outputs. MinerU is run externally on a GPU server, and the Tabulus library reads the resulting structured output to expose table regions, image paths, page numbers, bounding boxes, captions, footnotes, and MinerU `table_body` values.

## How To Use This Documentation

Start with the GPU and library setup pages if you are preparing an environment. Then follow the tutorial in order, beginning with PDF profiling.

For the current implementation, the most important distinction is:

- **Already validated:** MinerU 3.4.5 execution on GPU and typed table-region discovery with `tabulus.mineru`.
- **Next stages:** table-crop export, `tables_index.json` generation, PaddleOCR-VL execution, reference processing, and full end-to-end commands.

## Where To Start

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Set Up The Environment
:link: installation/gpu-server
:link-type: doc

Prepare the GPU-server workflow, separate Conda environments, and MinerU installation.
:::

:::{grid-item-card} Install The Python Library
:link: installation/python-library
:link-type: doc

Install Tabulus as a Python library and run GPU-independent unit tests.
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

- locate MinerU `*_content_list.json` files
- parse document elements
- select table regions
- resolve MinerU-generated table images
- convert zero-based MinerU page indices into document page numbers
- retain provenance and expose typed `TableRegion` objects

The current library does not yet launch MinerU, export table crops into the final handoff directory, run PaddleOCR-VL, run bibliography/reference matching, or produce final CSV outputs.

## Documentation Map

The sidebar contains the full documentation. The main sections are:

- **Installation And Setup:** GPU server and Python library setup.
- **Tutorial:** the intended modular workflow, one processing step at a time.
- **Components:** adapter boundaries and responsibilities.
- **Workflows:** headless local/GPU execution shapes.
- **Data Contracts:** the file formats exchanged between modules.
- **Evaluation:** how to compare extraction and matching quality.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Installation And Setup

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
