# Current State

Tabulus is being reorganized into an installable Python library using a standard `src/tabulus` package structure.

The first implemented library module provides MinerU-backed PDF profiling utilities. It can launch MinerU through the `tabulus profile` command, and it can also inspect an existing MinerU document-specific output directory.

## Validated Library Behavior

The current `tabulus.mineru` module:

- selects a CPU-compatible or GPU-backed MinerU backend
- constructs and runs non-interactive MinerU commands
- writes MinerU stdout, stderr, and run metadata logs
- recursively locates MinerU `*_content_list.json` files
- loads MinerU's structured content representation
- identifies entries where `type == "table"`
- resolves each table entry's associated image file
- converts MinerU's zero-based `page_idx` values into document page numbers
- preserves bounding boxes, captions, footnotes, and `table_body`
- optionally marks table regions that occur after a detected bibliography heading
- exposes typed `TableRegion` objects

The current `tabulus export-table-crops` command:

- consumes an existing MinerU output directory
- copies only discovered table images into an `images/` handoff directory
- preserves the source image extension
- writes a normalized `tables_index.json`
- keeps MinerU provenance, including original `img_path`, source image path, page number, bounding box, captions, footnotes, `table_body`, and reference-section status

The behavior is covered by unit tests that do not require GPU execution.

The current tests verify:

- content-list discovery
- table extraction
- page and provenance handling
- reference-section detection
- missing-output error handling
- backend selection and MinerU command construction
- mocked MinerU execution logging
- table-crop export and missing-source-image errors

## Validated Execution

MinerU 3.4.5 has been tested through two paths:

- Windows 11 CPU-only setup with Python 3.12, CPU-only PyTorch 2.10.0+cpu, and MinerU `pipeline`
- Linux GPU-server setup with Python 3.12, a dedicated Conda environment, and MinerU `hybrid-engine`

The tested GPU workflow is:

```text
PDF
  |
  v
MinerU 3.4.5
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

For the tested 53-page Puurunen 2005 GPU run, the library found 23 tables. Detected table regions began on page 6 and ended on page 22.

On Windows, the validated CPU-only state was Python 3.12.x, MinerU 3.4.5, PyTorch 2.10.0+cpu, and `torch.cuda.is_available()` returning `False`.

## Not Yet Implemented In The New Library

- PaddleOCR-VL execution
- GROBID, Kreuzberg, or Crossref integration
- full Tabulus process command

## Legacy Code Areas

The repository still contains legacy and research-oriented code:

- `src/legacy_tabulus`: older production-oriented pipeline and services
- `src/ocr_models`: OCR services, model experiments, and benchmark runners
- `evaluation`: evaluation scripts, plots, and result summaries

The documentation should keep the new library boundary separate from these legacy service and benchmark areas.

## Known Documentation Drift

- Some READMEs mention `tabulus/pipeline`, but the current legacy tree is `src/legacy_tabulus`.
- Some READMEs mention `paddle_service`, but the actual legacy folder is `paddleocr_service`.
- Docker instructions are stale for Windows machines without NVIDIA GPUs.
- Legacy service requirements remain separate from the minimal root development install contract.
