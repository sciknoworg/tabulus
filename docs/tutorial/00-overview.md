# Core Pipeline Overview

Tabulus extracts structured table data from scientific PDFs while keeping each processing stage inspectable on disk. The rebuilt library is organized around standalone commands and file contracts rather than one monolithic runner.

The current pipeline does not yet end in DOI-enriched final CSVs. It currently supports PDF profiling, canonical table-crop export, table reconstruction, reference-table classification, GROBID-backed bibliography extraction, and deterministic reference matching. The bibliography branch starts from the original PDF in parallel with table processing; DOI resolution, resolved CSV export, run reports, and complete `tabulus run` orchestration remain planned for the rebuilt library.

## Current Runnable Pipeline

The current command-line stages are:

1. **PDF Profiling:** `tabulus profile`
2. **Table Reconstruction:** `tabulus reconstruct-tables`
3. **Reference-Table Classification:** `tabulus classify-reference-tables`

Stage 4 bibliography extraction is implemented as a Python library API under
`src/tabulus/bibliography/`; a `tabulus` CLI subcommand for this stage is not
yet exposed.

The stage boundaries are persisted as files:

```text
PDF
  |
  v
MinerU native output
  |
  v
canonical table-crops/<paper>/
  |-- tables_index.json
  `-- images/
        |
        v
reconstructions/<adapter>/
  |-- native/
  |-- parsed/
  |-- predictions/
  `-- batch_summary.json
        |
        v
reference_table_classification.json

PDF
  |
  v
<artifact-root>/references/bibliography.json
```

`predictions/*.csv` files are reconstruction outputs before reference resolution. They are not bibliography-enriched or DOI-resolved final CSVs.

## Artifact Flow

```text
Scientific PDF
      |
      +--> MinerU / PDF Profiling
      |         |
      |         +--> MinerU table_body -------------------+
      |         |                                         |
      |         +--> canonical table crops                 |
      |                   |                               |
      |                   +--> crop-consuming adapters     |
      |                   |    (OCR, document VLM,         |
      |                   |     table-structure, or hybrid routes)
      |                   v                               |
      |         adapter-native reconstruction evidence     |
      |                   |                               |
      |                   v                               |
      |         shared structural parsing / normalization  |
      |                   |                               |
      |                   +-------------------------------+
      |                                   |
      |                                   v
      |                         reconstruction candidates
      |                                   |
      |                                   v
      |                           prediction CSVs
      |                                   |
      |                  +----------------+----------------+
      |                  |                                 |
      |                  v                                 v
      |          reconstruction evaluation      reference-table classification
      |
      +--> GROBID bibliography extraction
                |
                v
          references/bibliography.json

reference-table classification + bibliography.json
      |
      v
references/reference_matches.json
      |
      v
DOI resolution -> resolved CSV (planned)
```

MinerU is the current PDF profiler. It performs document/layout processing, table localization, and native table extraction. Tabulus reads MinerU output, exports the canonical table-crop handoff, and retains MinerU `table_body` as a native reconstruction candidate.

The crop-consuming reconstruction adapters currently registered in the rebuilt
library are listed in {doc}`08-table-ocr`. Each adapter receives the same
canonical MinerU crop; adapters must not independently locate or recrop tables
from the source PDF for the reconstruction comparison.

During reconstruction, adapter-native output is preserved under `native/`, then parsed through the shared Tabulus table parser into `parsed/`. A prediction CSV is written under `predictions/` only when exactly one usable parsed table is available for the physical crop.

Reference-table classification consumes reconstruction artifacts and writes `reference_table_classification.json` beside them. It does not overwrite raw reconstruction predictions.

Bibliography extraction is a separate PDF-level branch. It reads the original scientific PDF and writes normalized entries to `references/bibliography.json`; it does not consume canonical table crops or reconstruction prediction CSVs. The table and bibliography branches converge at deterministic reference matching.

## Current Versus Planned

Implemented in the rebuilt library:

- MinerU profiling through `tabulus profile`
- automatic canonical table-crop export
- standalone crop export through `tabulus export-table-crops`
- table reconstruction through `tabulus reconstruct-tables`
- registered crop-consuming reconstruction adapters listed in
  {doc}`08-table-ocr`
- shared HTML/Markdown structural parsing and deterministic OTSL-to-HTML normalization during reconstruction
- reference-table classification through `tabulus classify-reference-tables`
- GROBID-backed bibliography extraction through `src/tabulus/bibliography/`
- deterministic reference matching from selected reference-like tables and
  `references/bibliography.json`

Planned for the rebuilt library:

- DOI resolution
- resolved CSV export
- run report / QA bundle
- complete `tabulus run` orchestration

## Detailed Pages

- {doc}`01-pdf-profiling`
- {doc}`08-table-ocr`
- {doc}`10-reference-table-classification`
- {doc}`11-bibliography-extraction`
- {doc}`../modules/table-ocr-adapters`
- {doc}`../data-contracts/run-directory`
- {doc}`../external-tools/mineru`
- External Tools pages for adapter-specific model and runtime details
