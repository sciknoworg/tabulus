# Core Pipeline Overview

Tabulus extracts structured table data from scientific PDFs while keeping each processing stage inspectable on disk. The rebuilt library is organized around standalone commands and file contracts rather than one monolithic runner.

The current pipeline does not yet end in DOI-enriched final CSVs. It currently supports PDF profiling, canonical table-crop export, table reconstruction, reference-table classification, GROBID-backed bibliography extraction, deterministic reference matching, and paper-level scholarly reference resolution. The bibliography branch starts from the original PDF in parallel with table processing; Stage 7 resolved export, run reports, and complete `tabulus run` orchestration remain planned for the rebuilt library.

## Current Runnable Pipeline

The implemented pipeline runs through Stage 6 in the rebuilt `src/tabulus`
package:

1. **PDF Profiling:** `tabulus profile`
2. **Table Reconstruction:** `tabulus reconstruct-tables`
3. **Reference-Table Classification:** `tabulus classify-reference-tables`
4. **Bibliography Extraction:** `tabulus extract-bibliography`
5. **Reference Matching:** `tabulus match-references`
6. **Scholarly Reference Resolution:** `tabulus resolve-references`

Stage 7 resolved export is not implemented in the rebuilt package in this
checkout.

## Canonical TabulusBench Example

When a concrete worked example is needed, the tutorial uses TabulusBench paper
`P4`:

```text
Biomedicine_And_Health/clinical_research/P4/
  P4.pdf
  reference_tables/
  bibliography/gold.json
```

Set a portable benchmark root before running examples:

```bash
export TABULUSBENCH_ROOT="/path/to/tabulusbench"
export TABULUS_WORK="/path/to/tabulus-work"
P4_PDF="$TABULUSBENCH_ROOT/Biomedicine_And_Health/clinical_research/P4/P4.pdf"
```

In this documentation, a one-paper run means processing the complete relevant
input for one paper. For `P4`, PDF-level stages operate on `P4.pdf`; table-level
benchmark examples should use all six annotated reference-containing table
inputs when benchmark crops are the appropriate input; paper-level stages
operate on complete paper-level artifacts derived from `P4`.

The benchmark's `P4/reference_tables/` directory is human gold material. It
contains six annotated reference-containing tables with adjacent `gold.csv`
files. Those tables are not necessarily every table that Stage 1 profiling
will detect in the original PDF. Tabulus runs should write their own profiling
and crop artifacts outside the benchmark gold directories.

A full TabulusBench run means processing the complete applicable benchmark
input across all papers. Because TabulusBench papers are nested under
domain/subdomain directories, use an explicit `--pdf-list` for PDF-level stages
rather than `--folder` on the benchmark root. Running a stage over TabulusBench
is separate from evaluating it against gold annotations: for example,
bibliography extraction can be run for every benchmark PDF even though curated
bibliography gold exists only for a subset.

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
Stage 6: references/reference_resolution.json
  |
  v
Stage 7: join resolved identities to all relevant cells / export (planned)
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
Stage 6: references/reference_resolution.json
      |
      v
Stage 7: join resolved identities to all relevant cells / export (planned)
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
Stage 6 consumes the union of matched bibliography indices for a paper and
avoids resolving the same bibliography entry separately for every cell, table,
or reconstruction adapter. It writes a paper-level
`references/reference_resolution.json` registry after every target reaches a
final scientific status.

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
- reference matching through `tabulus match-references`
- paper-level scholarly reference resolution through `tabulus resolve-references`

Planned for the rebuilt library:

- resolved CSV export by joining resolved identities back onto Stage 5 table
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
