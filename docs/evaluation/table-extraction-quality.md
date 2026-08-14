# Table Extraction Quality

Table extraction quality evaluates whether detected and OCR-extracted tables match ground truth tables.

Current metrics include RMS-based similarity, normal accuracy, F1 score, and runtime depending on the evaluation script.

## First Comparison

The first table extraction comparison should test:

```text
MinerU table_body

versus

MinerU crop -> PaddleOCR-VL reconstruction
```

This comparison determines whether PaddleOCR-VL improves table reconstruction enough to justify the additional model step for a given table class.

## Runtime Context

When reporting extraction quality, record runtime context alongside accuracy metrics. MinerU first-run timings can include model download, vLLM startup, Torch compilation, CUDA graph capture, and cache warm-up. Those costs should be separated from steady-state document processing time.

For reproducible comparisons, record:

- document page count
- number of detected tables
- MinerU version, backend, and effort setting
- PaddleOCR-VL and PaddleOCR versions
- GPU model and number of visible GPUs
- first-run or warmed-cache status
- total wall-clock time per stage
- peak GPU memory if available
