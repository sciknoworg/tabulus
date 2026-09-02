# Python Library Setup

The new Tabulus code is organized as an installable Python library with a standard `src/tabulus` package layout.

Use this setup for core library development and unit tests that do not require GPU execution. For a validated Windows CPU-only MinerU installation, use `installation/windows-cpu`. For GPU-accelerated MinerU profiling, use `installation/gpu-server`.

## Development Install

From the repository root:

```bash
python -m pip install -e .
```

If you intend to run the test suite, install the development extra:

```bash
python -m pip install -e ".[dev]"
```

After installation, verify the MinerU output reader can be imported:

```bash
python - <<'PY'
from tabulus.mineru import discover_tables

print(discover_tables)
PY
```

## Unit Tests

The implemented library is designed so most unit tests do not require GPU access. Tests cover MinerU output discovery, backend selection, command construction, mocked MinerU execution, automatic table-crop export, standalone table-crop export, the table reconstruction adapter registry, PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, Granite Vision, TRivia, GLM-OCR, Dolphin-v2, DeepSeek-OCR-2, Nanonets-OCR-s, MonkeyOCRv2-B-Parsing, NVIDIA Nemotron Parse v1.2, HunyuanOCR-1.5, and dots.mocr adapter behavior, legacy-compatible table parsing, OTSL normalization, batch table reconstruction, and table reconstruction output writing.

Run the tests with:

```bash
python -m pytest
```

The current tests cover:

- recursive `*_content_list.json` discovery
- table-region extraction
- page and provenance handling
- reference-section detection
- missing-output error handling
- profile-driven automatic table-crop export
- table reconstruction registry lazy loading
- PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, Granite Vision, TRivia, GLM-OCR, Dolphin-v2, DeepSeek-OCR-2, Nanonets-OCR-s, MonkeyOCRv2-B-Parsing, NVIDIA Nemotron Parse v1.2, HunyuanOCR-1.5, and dots.mocr adapter behavior with mocked dependencies
- HTML-first, Markdown-fallback table parsing and deterministic OTSL-to-HTML normalization
- batch reconstruction input loading and output dispatch
- native, parsed, prediction CSV, and batch-summary artifact writing

## Current Import Boundary

The current public entry point is:

```python
from pathlib import Path
from tabulus.mineru import discover_tables

tables, refs_start_page = discover_tables(Path("work/mineru/puurunen_2005"))
```

This call inspects a MinerU document output directory. To create that output through Tabulus, use `tabulus profile` in an environment where MinerU is installed.

## Current CLI Commands

After installation, these commands should be available:

```bash
tabulus --version
tabulus profile --help
tabulus export-table-crops --help
tabulus reconstruct-tables --help
```

`tabulus profile` can launch MinerU when MinerU is installed in the active environment. The default `pipeline` backend is CPU-compatible; `hybrid-engine` is selected only when requested and a suitable CUDA GPU is visible.

If `hybrid-engine` is requested but the GPU requirements are not satisfied, Tabulus reports the reason and falls back to `pipeline`.

The profiling CLI separates the profiler from its backend. `mineru` is currently the only profiler; `pipeline` and `hybrid-engine` are MinerU backends.

When `--out` is omitted, Tabulus writes profiling output beside the PDF:

```text
<PDF directory>/tabulus-output/<profiler>/<backend>/
```

`--out` remains available as an explicit override. If `hybrid-engine` falls back to `pipeline`, the automatic directory uses the resolved backend, `pipeline`.

After a successful profiling run, `tabulus profile` exports canonical MinerU table crops automatically by default:

```text
<PDF directory>/tabulus-output/table-crops/<PDF stem>/
  tables_index.json
  images/
```

Use `--table-crops-out PATH` to override that handoff directory, or `--no-export-table-crops` to skip automatic crop export.

`tabulus export-table-crops` remains useful for regenerating the normalized handoff from an existing MinerU output directory without rerunning MinerU:

```text
work/table_crops/
  tables_index.json
  images/
```

The table reconstruction adapter package is available as `tabulus.table_ocr`. The registered crop-consuming reconstruction adapters are `paddleocr-vl`, `chandra`, `nuextract3`, `tesseract-tatr`, `rapidocr-tableformer`, `granite-vision-table`, `trivia`, `glm-ocr`, `dolphin-v2`, `deepseek-ocr-2`, `nanonets-ocr-s`, `monkeyocrv2-b-parsing`, `nemotron-parse-v1-2`, `hunyuanocr-1-5`, and `dots-mocr`. Their ML dependencies are optional and loaded only when the selected adapter is instantiated.

`tabulus reconstruct-tables` runs one registered table-reconstruction adapter across every crop in a canonical `tables_index.json` handoff. Registered crop-consuming adapters include `paddleocr-vl`, `chandra`, `nuextract3`, `tesseract-tatr`, `rapidocr-tableformer`, `granite-vision-table`, `trivia`, `glm-ocr`, `dolphin-v2`, `deepseek-ocr-2`, `nanonets-ocr-s`, `monkeyocrv2-b-parsing`, `nemotron-parse-v1-2`, `hunyuanocr-1-5`, and `dots-mocr`:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter paddleocr-vl \
  --device gpu:0
```

If `--out` is omitted, the command writes to:

```text
<crop-root>/reconstructions/<adapter>/
  native/
  parsed/
  predictions/
  batch_summary.json
```

This command writes prediction CSV files before reference resolution. It does not run bibliography matching, DOI enrichment, final resolved CSV export, or the complete end-to-end pipeline.
