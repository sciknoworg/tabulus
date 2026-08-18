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

## Extended Adapter Comparison

Later benchmarking can evaluate six candidate table reconstructions against the same ground-truth table:

- MinerU `table_body`
- MinerU crop -> PaddleOCR-VL
- MinerU crop -> DeepSeek OCR
- MinerU crop -> Chandra
- MinerU crop -> Kreuzberg
- MinerU crop -> NuExtract3

MinerU `table_body` is produced during PDF profiling. The other table-reconstruction adapters consume the same MinerU-generated table crop through the normalized table-crop handoff. They should not independently process the original PDF to detect tables, set bounding boxes, or create competing crops for this comparison.

```text
same detected table
        |
        +-- MinerU table_body
        |
        +-- MinerU crop
              +-- PaddleOCR-VL
              +-- DeepSeek OCR
              +-- Chandra
              +-- Kreuzberg
              +-- NuExtract3
```

Because the adapter-native output formats may differ, Tabulus should normalize every candidate into a common table representation before scoring against the ground-truth CSV.

This comparison should not assume a winner. It should measure whether an external adapter improves over MinerU's own `table_body`, and which adapter offers the best quality/runtime tradeoff for different table classes. Using the same MinerU crop controls the table-detection and cropping variable, so the evaluation is primarily about table reconstruction quality.

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
