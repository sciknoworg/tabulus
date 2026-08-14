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
