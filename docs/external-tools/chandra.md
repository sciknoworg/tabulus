# Chandra OCR 2

Chandra OCR 2 is an external OCR model used by Tabulus for table reconstruction from canonical MinerU table crops.

## Official Resources

- Chandra OCR project repository: [akanugan/chandra-ocr](https://github.com/akanugan/chandra-ocr)
- Exact model used by Tabulus: [datalab-to/chandra-ocr-2](https://huggingface.co/datalab-to/chandra-ocr-2)

## Role In Tabulus

Tabulus exposes Chandra through the current table-reconstruction adapter contract only. For the generic adapter architecture and output contract, see {doc}`../modules/table-ocr-adapters`.

Chandra is registered as:

```text
--adapter chandra
```

The adapter is implemented in `src/tabulus/table_ocr/chandra.py`. It consumes one canonical MinerU crop at a time and does not redetect tables or recrop the original PDF.

The implemented path is:

```text
canonical MinerU crop
      |
      v
Chandra OCR 2
      |
      v
raw structured HTML
      |
      v
Tabulus common span-aware HTML parser
      |
      v
parsed rectangular representation
      |
      v
prediction CSV
```

## Invocation

Run Chandra through the shared reconstruction CLI:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter chandra \
  --device gpu:0
```

For multiple papers:

```bash
tabulus reconstruct-tables \
  --crops-folder "/path/to/tabulus-output/table-crops" \
  --adapter chandra \
  --device gpu:0
```

The default output is:

```text
<crop-root>/
  reconstructions/
    chandra/
      native/
      parsed/
      predictions/
      batch_summary.json
```

## Settings Used By Tabulus

The current adapter uses:

- package: `chandra-ocr` 0.2.0
- model: `datalab-to/chandra-ocr-2`
- backend: Hugging Face/in-process API
- prompt: `prompt_type="ocr"`

Tabulus maps `--device gpu:0` to PyTorch `cuda:0`. A Chandra model instance is loaded lazily and reused by the batch layer across crops.

Tabulus does not require the Chandra CLI or a vLLM server for the implemented adapter path.

## Native Output

Chandra returns generated HTML plus metadata. Tabulus preserves the raw generated content, token count, and Chandra error status in the native adapter result. The raw HTML is then parsed by the shared span-aware HTML/Markdown parser; there is no Chandra-specific Tabulus table parser.

## Validated Configuration

The validated Chandra stack used:

- Python 3.12.13
- `chandra-ocr` 0.2.0
- model `datalab-to/chandra-ocr-2`
- PyTorch 2.13.0+cu130
- Transformers 5.15.1
- NVIDIA L40S
- `prompt_type="ocr"`

A real `tabulus reconstruct-tables --adapter chandra --device gpu:0` smoke test against one canonical MinerU crop completed with status `ok`, one result, one structured HTML table, a 65 x 6 parsed representation, one prediction CSV, token count 3838, and Chandra generation error `false`.

This smoke test demonstrates adapter integration, not accuracy superiority over another reconstruction model.
