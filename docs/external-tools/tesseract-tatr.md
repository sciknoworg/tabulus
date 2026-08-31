# Tesseract + Table Transformer

Tesseract + Table Transformer is an external-tool combination used by Tabulus
for table reconstruction from canonical MinerU table crops.

## Official Resources

- Tesseract OCR project repository: [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
- Microsoft Table Transformer project repository: [microsoft/table-transformer](https://github.com/microsoft/table-transformer)
- Exact TATR model used by Tabulus: [microsoft/table-transformer-structure-recognition-v1.1-all](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all)

## Role In Tabulus

Tabulus exposes this integration through the current table-reconstruction
adapter contract only. For the generic adapter architecture and output
contract, see {doc}`../modules/table-ocr-adapters`.

The adapter is registered as:

```text
--adapter tesseract-tatr
```

The display name is:

```text
Tesseract + Table Transformer
```

The adapter is implemented in `src/tabulus/table_ocr/tesseract_tatr.py`, with
deterministic structure postprocessing in
`src/tabulus/table_ocr/tatr_postprocess.py`. It consumes one canonical MinerU
crop at a time and does not redetect tables or recrop the original PDF.

The implemented path is:

```text
canonical MinerU crop
      |
      v
Tesseract OCR word tokens and bounding boxes
      |
      v
Microsoft Table Transformer structure recognition
      |
      v
deterministic token/structure fusion
      |
      v
HTML table
      |
      v
Tabulus common HTML/Markdown parser
      |
      v
parsed rectangular representation
      |
      v
prediction CSV
```

This adapter replaces the old Kreuzberg/Xberg reconstruction candidate with
explicitly named underlying components. Kreuzberg/Xberg is not an active
reconstruction adapter in the rebuilt library.

## Invocation

Run Tesseract + Table Transformer through the shared reconstruction CLI:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter tesseract-tatr \
  --device gpu:0
```

For multiple papers:

```bash
tabulus reconstruct-tables \
  --crops-folder "/path/to/tabulus-output/table-crops" \
  --adapter tesseract-tatr \
  --device gpu:0
```

The default output is:

```text
<crop-root>/
  reconstructions/
    tesseract-tatr/
      native/
      parsed/
      predictions/
      batch_summary.json
```

## Settings Used By Tabulus

The current adapter uses:

- Tesseract language: `eng`
- Tesseract page segmentation mode: `--psm 6`
- Table Transformer model: `microsoft/table-transformer-structure-recognition-v1.1-all`
- TATR object threshold: `0.5`
- TATR max image dimension: `1000`
- preprocessing: resize with max dimension 1000, convert to tensor, and apply
  ImageNet normalization

Tabulus maps `--device cpu` to PyTorch `cpu`, `--device gpu` to PyTorch
`cuda`, and `--device gpu:<index>` to `cuda:<index>`.

The adapter requires the Tesseract executable on `PATH` plus PyTorch,
torchvision, Transformers 4.x, Pillow, timm, and PyMuPDF in the active
environment. The committed adapter explicitly rejects Transformers 5.x for the
validated model configuration.

## Native Output

Tabulus preserves adapter-native evidence in `native/`, including:

- Tesseract version, language, page segmentation mode, command, OCR word
  tokens, and word bounding boxes
- TATR model id, object threshold, max image size, detected objects,
  row/column/header structure, fused cells, and token-slot confidence

The deterministic postprocessing produces an HTML table, which is stored as the
adapter's native Markdown/HTML view and then parsed by the shared Tabulus
HTML/Markdown parser. There is no reference-resolution logic in this adapter.

## Validated Configuration

The adapter has been validated through the real `tabulus reconstruct-tables`
CLI on one scientific canonical crop and on one selected three-document
engineering-validation slice used during development.

For that selected slice:

```text
canonical crops: 83
status ok:       83
prediction CSVs: 83
empty:           0
errors:          0
runtime:         approximately 165.78 s
hardware:        NVIDIA L40S setup
```

The 83-crop count is specific to that selected engineering-validation slice. It
is not a benchmark size, software property, or accuracy result. Runtime and
prediction yield are operational observations only and do not show that this
adapter is more accurate than PaddleOCR-VL, Chandra OCR 2, NuExtract3, or any
future adapter.

Scientific comparison must use gold-standard evaluation.

## Boundary

This adapter performs table reconstruction only. It does not perform
original-PDF table redetection, recropping, continued-table merging,
reference-table classification, reference-resolution heuristics, DOI
resolution, or final resolved CSV generation.
