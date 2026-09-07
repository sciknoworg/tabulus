# Core Pipeline Overview

Tabulus extracts structured table data from scientific PDFs while keeping each processing stage inspectable on disk. The rebuilt library is organized around standalone commands and file contracts rather than one monolithic runner.

The current pipeline does not yet end in DOI-enriched final CSVs. It currently supports PDF profiling, canonical table-crop export, table reconstruction, reference-table classification, GROBID-backed bibliography extraction, deterministic reference matching, and Stage 6 paper-level scholarly reference resolution. The bibliography branch starts from the original PDF in parallel with table processing; Stage 7 resolved export, run reports, and complete `tabulus run` orchestration remain planned for the rebuilt library.

## Current Runnable Pipeline

The implemented pipeline runs through Stage 6. Commands shown for Stages 1-5
are verified in this checkout:

1. **PDF Profiling:** `tabulus profile`
2. **Table Reconstruction:** `tabulus reconstruct-tables`
3. **Reference-Table Classification:** `tabulus classify-reference-tables`
4. **Bibliography Extraction:** `tabulus extract-bibliography`
5. **Reference Matching:** `tabulus match-references`
6. **Paper-Level Scholarly Reference Resolution:** one
   `references/reference_resolution.json` registry per paper

Stage 6 scholarly reference resolution is implemented on the GPU cluster.
See {doc}`../project-notes/current-state` for the local checkout boundary and
{doc}`13-doi-resolution` for its workflow. Stage 7 is not implemented.

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
selected_reference_tables.json (Stage 3)

PDF
  |
  v
<artifact-root>/references/bibliography.json (Stage 4)

selected_reference_tables.json + bibliography.json
  |
  v
references/reference_matches.json (Stage 5; table-cell links)
  |
  v
union + dedupe of matched bibliography indices across all methods
  |
  v
references/reference_resolution.json (Stage 6; one registry per paper)
  |
  v
Stage 7: join validated identities to all relevant cells / export (planned)
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

selected_reference_tables.json + bibliography.json
      |
      v
references/reference_matches.json
      |
      v
union + dedupe of referenced bibliography indices across methods
      |
      v
references/reference_resolution.json (Stage 6; one registry per paper)
      |
      v
Stage 7: join identities to all relevant cells / export (planned)
```

MinerU is the current PDF profiler. It performs document/layout processing, table localization, and native table extraction. Tabulus reads MinerU output, exports the canonical table-crop handoff, and retains MinerU `table_body` as a native reconstruction candidate.

The crop-consuming reconstruction adapters currently registered in the rebuilt
library are listed in {doc}`08-table-ocr`. Each adapter receives the same
canonical MinerU crop; adapters must not independently locate or recrop tables
from the source PDF for the reconstruction comparison.

During reconstruction, adapter-native output is preserved under `native/`, then parsed through the shared Tabulus table parser into `parsed/`. A prediction CSV is written under `predictions/` only when exactly one usable parsed table is available for the physical crop.

One deterministic regex/rule classifier is applied independently to each
reconstruction method's outputs. It identifies reference-containing
reconstructed-table instances and writes `reference_table_classification.json` beside them. It does not overwrite raw reconstruction predictions.

Bibliography extraction is a separate PDF-level branch. It reads the original scientific PDF and writes normalized entries to `references/bibliography.json`; it does not consume canonical table crops or reconstruction prediction CSVs. The table and bibliography branches converge at deterministic reference matching.

Stage 5 links table-cell citation tokens to bibliography positions offline.
Stage 6 resolves the union of unique bibliography entries referenced by
the selected tables across all reconstruction methods of a paper, once per `(paper, bibliography_index)`, rather
than resolving references separately for every cell, table, or reconstruction
adapter. This keeps scholarly identity decisions paper-level and prevents
repeated external resolution calls for the same bibliography entry.

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
- GROBID-backed bibliography extraction through `tabulus extract-bibliography`
- deterministic reference matching from selected reference-like tables and
  `references/bibliography.json`
- Stage 6 scholarly resolution as one paper-level
  `references/reference_resolution.json` registry

Planned for the rebuilt library:

- resolved CSV export by joining the Stage 6 registry back onto Stage 5 table
  links
- run report / QA bundle
- complete `tabulus run` orchestration

## Detailed Pages

- {doc}`01-pdf-profiling`
- {doc}`08-table-ocr`
- {doc}`10-reference-table-classification`
- {doc}`11-bibliography-extraction`
- {doc}`12-reference-matching`
- {doc}`13-doi-resolution`
- {doc}`14-csv-export`
- {doc}`../modules/table-ocr-adapters`
- {doc}`../data-contracts/run-directory`
- {doc}`../external-tools/mineru`
- External Tools pages for adapter-specific model and runtime details
