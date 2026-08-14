# Tabulus

Tabulus is a modular pipeline for digitizing scientific PDF papers into structured, reference-aware table data.

The documentation is organized around processing components, not around specific OCR or extraction libraries. Each component should be runnable on its own, exchange data through a documented contract, and plug into the end-to-end pipeline when the individual step is stable.

```{toctree}
:maxdepth: 2
:caption: Installation And Setup

installation/gpu-server
```

```{toctree}
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
:maxdepth: 2
:caption: Workflows

workflows/single-pdf-local
workflows/gpu-server-run
workflows/end-to-end-run
workflows/debugging-failed-step
```

```{toctree}
:maxdepth: 2
:caption: Data Contracts

data-contracts/run-directory
data-contracts/pdf-profile-json
data-contracts/tables-index-json
data-contracts/ocr-tables-json
data-contracts/bibliography-json
data-contracts/reference-matches-json
data-contracts/resolved-csv
```

```{toctree}
:maxdepth: 2
:caption: Evaluation

evaluation/overview
evaluation/table-extraction-quality
evaluation/bibliography-quality
evaluation/reference-matching-quality
```

```{toctree}
:maxdepth: 1
:caption: Project Notes

project-notes/current-state
project-notes/branding
project-notes/containerization-later
```
