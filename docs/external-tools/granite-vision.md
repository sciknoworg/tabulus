# Granite Vision 4.1 4B

Granite Vision 4.1 4B is a vision-language model used by Tabulus for table
reconstruction from canonical MinerU table crops. It is not a conventional OCR
engine: the model generates table structure and cell text together as OTSL.

## Official Resources

- [Granite Vision 4.1 4B model](https://huggingface.co/ibm-granite/granite-vision-4.1-4b)
- [Docling project repository](https://github.com/docling-project/docling)

## Role In Tabulus

The registered Tabulus adapter is:

```text
granite-vision-table
```

It receives the same canonical MinerU crop used by the other crop-consuming
reconstruction adapters. The image is sent directly to Granite Vision; Tabulus
does not run Docling PDF conversion, page-layout detection, table detection,
redetection, or candidate-specific recropping.

The reconstruction path is:

```text
canonical MinerU crop
        |
        v
Granite Vision 4.1 4B
        |
        v
<tables_otsl> generation
        |
        v
Docling Granite OTSL parsing
        |
        v
structured cells
        |
        v
shared Tabulus parser/output contract
```

For the generic adapter interface and artifact contract, see
{doc}`../modules/table-ocr-adapters`.

## Invocation

Reconstruct one paper's canonical crops:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter granite-vision-table \
  --device gpu:0
```

The same adapter can be used with `--crops-folder` for a sequential
multi-paper batch.

## Settings Used By Tabulus

The validated integration uses:

- model: `ibm-granite/granite-vision-4.1-4b`
- model revision: `dd48e97503de471803850df70843cf9eb5da8712`
- task prompt: `<tables_otsl>`
- Docling: `2.123.1`
- Transformers: `4.57.3`
- dtype: `bfloat16`
- attention implementation: SDPA
- registry capability: GPU only

The model generates OTSL containing table structure and cell text. Tabulus uses
Docling's Granite OTSL parsing implementation to convert that output into
structured cells, then renders those cells into the shared table parser/output
contract. The adapter loads the model lazily and keeps each physical MinerU
crop independent.

## Native Output

The adapter preserves the Granite model and revision, raw generated output, OTSL
sequence, structured cells and table dimensions, image dimensions, device and
version information, and related generation provenance as native evidence:

```text
<crop-root>/
  reconstructions/
    granite-vision-table/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be `ok` and exactly one structured table must parse from the crop. Native
and parsed evidence is retained for empty or ambiguous results.

## Limitations

This adapter uses the complete canonical MinerU crop and does not independently
locate or crop tables from the original PDF. It does not use a separate OCR
engine, semantically correct cells, merge continued tables, apply reference-
resolution heuristics, extract bibliographies, match references, resolve DOI
values, or generate final resolved CSV files.

Validation has also found an input-dependent robustness limitation: some table
images can trigger pathological Granite Vision generation with very long
per-table runtimes and extreme CPU/GPU memory consumption. The adapter works
successfully on many tables, but the current implementation does not yet impose
an explicit per-table timeout or conservative generation ceiling. A pathological
input can therefore monopolize resources and prevent the remaining batch from
being processed.
