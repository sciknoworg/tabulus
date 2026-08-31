# Core Pipeline Overview

Tabulus extracts structured table data from scientific PDFs while keeping each processing stage inspectable on disk. The rebuilt library is organized around standalone commands and file contracts rather than one monolithic runner.

The current pipeline does not yet end in DOI-enriched final CSVs. It currently supports PDF profiling, canonical table-crop export, table reconstruction, and reference-table classification. Bibliography extraction, reference matching, DOI resolution, resolved CSV export, run reports, and complete `tabulus run` orchestration remain planned for the rebuilt library.

## Current Runnable Pipeline

The user-facing runnable stages are:

1. **PDF Profiling:** `tabulus profile`
2. **Table Reconstruction:** `tabulus reconstruct-tables`
3. **Reference-Table Classification:** `tabulus classify-reference-tables`

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
```

`predictions/*.csv` files are reconstruction outputs before reference resolution. They are not bibliography-enriched or DOI-resolved final CSVs.

## Artifact Flow

```text
Scientific PDF
      |
      v
MinerU / PDF Profiling
      |
      +--> MinerU table_body -----------------------------+
      |                                                   |
      +--> canonical table crop                           |
                |                                         |
                +--> PaddleOCR-VL                         |
                +--> Chandra OCR 2                        |
                +--> NuExtract3                           |
                +--> Tesseract + Table Transformer        |
                +--> RapidOCR + Docling TableFormer       |
                +--> Granite Vision 4.1 4B                |
                |                                         |
                v                                         |
      adapter-native reconstruction evidence              |
                |                                         |
                v                                         |
      shared structural parsing / normalization           |
                |                                         |
                +-----------------------------------------+
                                  |
                                  v
                        reconstruction candidates
                                  |
                                  v
                          prediction CSVs
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
         reconstruction evaluation      reference-table classification
                                                   |
                                                   v
                                      bibliography / matching /
                                      DOI resolution (planned)
                                                   |
                                                   v
                                           resolved CSV (planned)
```

MinerU is the current PDF profiler. It performs document/layout processing, table localization, and native table extraction. Tabulus reads MinerU output, exports the canonical table-crop handoff, and retains MinerU `table_body` as a native reconstruction candidate.

The crop-consuming reconstruction adapters currently registered in the rebuilt library are PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, and Granite Vision 4.1 4B. Each adapter receives the same canonical MinerU crop; adapters must not independently locate or recrop tables from the source PDF for the reconstruction comparison.

During reconstruction, adapter-native output is preserved under `native/`, then parsed through the shared Tabulus table parser into `parsed/`. A prediction CSV is written under `predictions/` only when exactly one usable parsed table is available for the physical crop.

Reference-table classification consumes reconstruction artifacts and writes `reference_table_classification.json` beside them. It does not overwrite raw reconstruction predictions.

## Current Versus Planned

Implemented in the rebuilt library:

- MinerU profiling through `tabulus profile`
- automatic canonical table-crop export
- standalone crop export through `tabulus export-table-crops`
- table reconstruction through `tabulus reconstruct-tables`
- registered PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, and Granite Vision 4.1 4B reconstruction adapters
- shared HTML/Markdown structural parsing during reconstruction
- reference-table classification through `tabulus classify-reference-tables`

Planned for the rebuilt library:

- DeepSeek OCR reconstruction adapter
- bibliography extraction
- reference matching
- DOI resolution
- resolved CSV export
- run report / QA bundle
- complete `tabulus run` orchestration

## Detailed Pages

- {doc}`01-pdf-profiling`
- {doc}`08-table-ocr`
- {doc}`10-reference-table-classification`
- {doc}`../modules/table-ocr-adapters`
- {doc}`../data-contracts/run-directory`
- {doc}`../external-tools/mineru`
- {doc}`../external-tools/paddleocr-vl`
- {doc}`../external-tools/chandra`
- {doc}`../external-tools/nuextract3`
- {doc}`../external-tools/tesseract-tatr`
- {doc}`../external-tools/docling`
- {doc}`../external-tools/granite-vision`
