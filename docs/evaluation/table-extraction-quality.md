# Table Extraction Quality

Table extraction quality evaluates whether reconstructed tables match ground truth tables.

Current evaluation scripts compare a reconstruction prediction CSV against a
ground-truth CSV. Metrics include RMS-based DePlot-style table similarity,
normal accuracy, F1 score, and runtime depending on the script.

```text
normalized reconstruction
        |
        v
prediction CSV
        |
        v
RMS / DePlot evaluation
        |
        v
ground-truth CSV
```

The DOI-enriched resolved CSV is not used for OCR/table reconstruction quality because enrichment intentionally changes reference-cell values.

Reference-table classification is a separate downstream task. Its quality
should be evaluated against classification labels when such labels are
available; do not mix classification accuracy with reconstruction quality.

## First Comparison

The first table extraction comparison should test:

```text
MinerU table_body

versus

MinerU crop -> PaddleOCR-VL reconstruction
```

This comparison determines whether PaddleOCR-VL improves table reconstruction enough to justify the additional model step for a given table class.

## Extended Adapter Comparison

Benchmarking can evaluate implemented and future candidate table reconstructions against the same ground-truth table:

- MinerU `table_body`
- MinerU crop -> PaddleOCR-VL
- MinerU crop -> Chandra
- MinerU crop -> NuExtract3
- MinerU crop -> Tesseract + Table Transformer
- MinerU crop -> RapidOCR + Docling TableFormer
- MinerU crop -> Granite Vision 4.1 4B
- MinerU crop -> DeepSeek OCR (future)

MinerU `table_body` is produced during PDF profiling. PaddleOCR-VL, Chandra, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, and Granite Vision 4.1 4B are current crop-consuming Tabulus reconstruction adapters. DeepSeek OCR remains future work. The crop-consuming adapters use the same MinerU-generated table crop through the normalized table-crop handoff. They should not independently process the original PDF to detect tables, set bounding boxes, or create competing crops for this comparison.

```text
same detected table
        |
        +-- MinerU table_body
        |
        +-- MinerU crop
              +-- PaddleOCR-VL
              +-- Chandra
              +-- NuExtract3
              +-- Tesseract + Table Transformer
              +-- RapidOCR + Docling TableFormer
              +-- Granite Vision 4.1 4B
              +-- DeepSeek OCR (future)
```

Because the adapter-native output formats may differ, Tabulus should normalize every candidate into a common table representation and export a prediction CSV before scoring against the ground-truth CSV.

This comparison should not assume a winner. It should measure whether an external adapter improves over MinerU's own `table_body`, and which adapter offers the best quality/runtime tradeoff for different table classes. Using the same MinerU crop controls the table-detection and cropping variable, so the evaluation is primarily about table reconstruction quality.

## Runtime Context

When reporting extraction quality, record runtime context alongside accuracy metrics. MinerU first-run timings can include model download, vLLM startup, Torch compilation, CUDA graph capture, and cache warm-up. Those costs should be separated from steady-state document processing time.

For reproducible comparisons, record:

- document page count
- number of detected tables
- MinerU version, backend, and effort setting
- reconstruction adapter name and model/runtime versions
- GPU model and number of visible GPUs
- first-run or warmed-cache status
- total wall-clock time per stage
- peak GPU memory if available

Structural subclass evaluation should group results by table category or difficulty class when curated labels are available. That makes it possible to see whether an adapter helps only on specific table structures rather than only reporting one aggregate score.
