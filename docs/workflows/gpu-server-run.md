# GPU Server Run

The GPU server workflow should run the same modules as the local workflow, but with GPU-backed adapters for layout detection and table OCR.

Before running this workflow, complete the GPU server setup in `installation/gpu-server`.

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

Then run:

```bash
python -m tabulus_pipeline.profile_manifest --manifest /data/papers.csv --runs-root /data/runs --adapter mineru
```

Each later module should process either one run or all runs with a selected status.

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

The GPU workflow should materialize a clean intermediate table-crop collection before invoking PaddleOCR-VL:

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
work/table_crops/
  table_index.json
  images/
```

This directory should preserve `page_idx`, `bbox`, captions, footnotes, `mineru_img_path`, and MinerU `table_body` when available. PaddleOCR-VL should receive the copied table image, not a full PDF page.
