# PaddleOCR-VL

PaddleOCR-VL is an external document vision-language model used by Tabulus for table reconstruction from canonical MinerU table crops.

Tabulus exposes PaddleOCR-VL only through the current table-reconstruction adapter contract. For the generic adapter architecture and output contract, see {doc}`../modules/table-ocr-adapters`.

## Role In Tabulus

PaddleOCR-VL is registered as:

```text
--adapter paddleocr-vl
```

The adapter is implemented in `src/tabulus/table_ocr/paddleocr_vl.py`. It consumes one canonical MinerU crop at a time and does not redetect tables or recrop the original PDF.

The workflow is:

```text
canonical MinerU crop
      |
      v
PaddleOCR-VL
      |
      v
native PaddleOCR result views
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

## Invocation

Run PaddleOCR-VL through the shared reconstruction CLI:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter paddleocr-vl \
  --device gpu:0
```

For multiple papers:

```bash
tabulus reconstruct-tables \
  --crops-folder "/path/to/tabulus-output/table-crops" \
  --adapter paddleocr-vl \
  --device gpu:0
```

The default output is:

```text
<crop-root>/
  reconstructions/
    paddleocr-vl/
      native/
      parsed/
      predictions/
      batch_summary.json
```

## Settings Used By Tabulus

The current adapter initializes PaddleOCR-VL with:

```python
PaddleOCRVL(
    pipeline_version="v1.6",
    device="<device>",
    engine="paddle",
    use_layout_detection=False,
)
```

For each crop, Tabulus calls prediction with:

```python
pipeline.predict(
    str(image_path),
    use_layout_detection=False,
    prompt_label="table",
)
```

`use_layout_detection=False` is intentional because MinerU has already localized and cropped the physical table. `prompt_label="table"` tells PaddleOCR-VL that the input image is already a table crop.

## Native Output

Tabulus preserves PaddleOCR-VL's public JSON and Markdown result views as adapter-native evidence in `native/`. The Markdown/HTML table representation is then parsed through the shared Tabulus parser and written to `parsed/`; a prediction CSV is written only when exactly one usable parsed table is found.

## Validated Configuration

Validated PaddleOCR-VL configurations include CPU and NVIDIA L40S GPU runs using PaddleOCR 3.7.0, PaddlePaddle 3.2.1, PaddleOCR-VL 1.6, `engine="paddle"`, and canonical MinerU crop input.

These validations demonstrate integration behavior. They are not scientific accuracy benchmarks and do not establish PaddleOCR-VL as preferable for every table class.
