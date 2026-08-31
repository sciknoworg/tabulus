# GPU Server Run

The GPU server workflow runs the same file-contract-oriented modules as the CPU workflow, but uses GPU-backed adapters where they are validated. For MinerU profiling, this means requesting `hybrid-engine` instead of the CPU-compatible `pipeline` backend. For table reconstruction, `tabulus reconstruct-tables` can run a registered crop-consuming adapter such as PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, or Granite Vision 4.1 4B over the canonical MinerU crop handoff in the appropriate adapter environment.

Before running this workflow, complete the GPU server setup in `installation/gpu-server`. For the exact tested MinerU command sequence, see `workflows/mineru-gpu-execution`.

## Assumptions

- Python environment is managed directly on the server.
- GPU libraries are installed outside Docker.
- Input PDFs are already available in a papers folder.
- Output runs are written to a persistent runs folder.

## Recommended Shape

Use a manifest when processing many papers:

```text
paper_id,pdf_path
P51,/data/papers/P51.pdf
P52,/data/papers/P52.pdf
```

Future full-pipeline manifest shape:

```bash
tabulus run --manifest /data/papers.csv --runs-root /data/runs
```

This full-manifest command is not yet implemented in the new library. The currently validated single-document commands are:

```bash
tabulus profile --pdf /data/papers/P51.pdf --backend hybrid-engine

tabulus reconstruct-tables \
  --crops /data/papers/tabulus-output/table-crops/P51 \
  --adapter paddleocr-vl \
  --device gpu:0
```

If `--out` is omitted, Tabulus uses the same default convention as the CPU workflow:

```text
<PDF directory>/tabulus-output/mineru/<resolved-backend>/
```

Do not pass `--out` unless a GPU server should intentionally write the profiling root to a particular shared work or runs directory.

When `hybrid-engine` is requested but unavailable, the resolved backend is `pipeline`, and the automatic directory uses `pipeline`.

After successful profiling, Tabulus automatically exports canonical MinerU table crops to:

```text
<PDF directory>/tabulus-output/table-crops/<PDF stem>/
```

The current rebuilt library implements reference-table classification after reconstruction with `tabulus classify-reference-tables`. Bibliography extraction, reference matching, DOI resolution, and resolved CSV export remain planned downstream stages.

## Profiling MinerU Runs

Separate first-run setup cost from document-processing cost. The first invocation may download the MinerU VLM checkpoint, initialize vLLM, compile Torch graphs, capture CUDA graphs, and download OCR/layout models. Those costs are cacheable and should not be mixed into steady-state per-document timing.

For each profiling run, record:

- MinerU version
- backend and effort setting
- document page count
- GPU model and number of visible GPUs
- model-loading and warm-up time
- layout-analysis time
- OCR-detection time
- OCR-recognition time
- total wall-clock time
- peak GPU memory if available
- number of detected tables

For a controlled benchmark, run the same document once to warm the environment and then run it again to estimate steady-state processing time.

## Table-Crop Handoff

The GPU workflow should materialize a clean intermediate table-crop collection before invoking a table reconstruction adapter:

```text
MinerU content_list.json
  |
  v
entries where type == "table"
  |
  v
table images resolved from img_path
  |
  v
tabulus-output/table-crops/<PDF stem>/
  tables_index.json
  images/
```

This directory should preserve `page_idx`, `bbox`, captions, footnotes, `mineru_img_path`, and MinerU `table_body` when available. A crop-consuming table-reconstruction adapter should receive the copied MinerU table image, not a full PDF page.

For batch reconstruction, the command writes one adapter-specific tree:

```text
tabulus-output/table-crops/<PDF stem>/
  reconstructions/
    <adapter>/
      native/
      parsed/
      predictions/
      batch_summary.json
```

These prediction CSV files remain pre-reference-resolution artifacts.
