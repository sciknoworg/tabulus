# Current State

Tabulus is being reorganized into an installable Python library using a standard `src/tabulus` package structure.

The first implemented library module provides typed access to existing MinerU document outputs. It does not run MinerU. It expects MinerU to have already produced a document-specific output directory.

## Validated Library Behavior

The current `tabulus.mineru` module:

- recursively locates MinerU `*_content_list.json` files
- loads MinerU's structured content representation
- identifies entries where `type == "table"`
- resolves each table entry's associated image file
- converts MinerU's zero-based `page_idx` values into document page numbers
- preserves bounding boxes, captions, footnotes, and `table_body`
- optionally marks table regions that occur after a detected bibliography heading
- exposes typed `TableRegion` objects

The behavior is covered by unit tests that do not require GPU execution.

The current tests verify:

- content-list discovery
- table extraction
- page and provenance handling
- reference-section detection
- missing-output error handling

## Validated External Execution

MinerU 3.4.5 has been tested on a GPU server in a dedicated Conda environment using Python 3.12.

The tested GPU workflow is:

```text
PDF
  |
  v
MinerU 3.4.5 on GPU
  |
  v
MinerU structured output
  |
  v
tabulus.mineru
  |
  v
typed TableRegion objects
```

For the tested 53-page Puurunen 2005 document, the library found 23 tables. Detected table regions began on page 6 and ended on page 22.

## Not Yet Implemented In The New Library

- MinerU process launching
- table JPG to PNG export
- `tables_index.json` generation
- PaddleOCR-VL execution
- GROBID, Kreuzberg, or Crossref integration
- full Tabulus process command

## Legacy Code Areas

The repository still contains legacy and research-oriented code:

- `src/Tabulus`: older production-oriented pipeline and services
- `src/ocr_models`: OCR services, model experiments, and benchmark runners
- `evaluation`: evaluation scripts, plots, and result summaries

The documentation should keep the new library boundary separate from these legacy service and benchmark areas.

## Known Documentation Drift

- Some READMEs mention `tabulus/pipeline`, but the current production folder in the legacy tree is `src/Tabulus`.
- Some READMEs mention `paddle_service`, but the actual legacy folder is `paddleocr_service`.
- Docker instructions are stale for Windows machines without NVIDIA GPUs.
- The root `requirements.txt` captures a broad research environment and should not be treated as the production install contract.
