# TRivia-3B

TRivia-3B is a vision-language model used by Tabulus for table reconstruction
from canonical MinerU table crops. It is not a conventional OCR engine and does
not use a separate table-detection step: the model receives the table crop and
generates native OTSL.

## Official Resources

- [TRivia-3B model](https://huggingface.co/opendatalab/TRivia-3B)

## Role In Tabulus

The registered Tabulus adapter is:

```text
trivia
```

It receives the same canonical MinerU crop used by the other crop-consuming
reconstruction adapters. Tabulus does not ask TRivia to redetect tables or
recrop the original PDF.

The reconstruction path is:

```text
canonical MinerU crop
        |
        v
TRivia-3B
        |
        v
native OTSL
        |
        v
Tabulus OTSL-to-HTML normalization
        |
        v
shared HTML parser
        |
        v
prediction CSV when exactly one table parses
```

For the generic adapter interface and artifact contract, see
{doc}`../modules/table-ocr-adapters`.

## Invocation

Run TRivia through the shared reconstruction CLI:

```bash
export CUDA_VISIBLE_DEVICES=0

tabulus reconstruct-tables \
  --crops /path/to/canonical/table-crops \
  --adapter trivia \
  --device gpu:0
```

The same adapter can be used with `--crops-folder` for a sequential
multi-paper batch.

## Settings Used By Tabulus

The validated integration uses:

- model: `opendatalab/TRivia-3B`
- model revision: `fcf890f3869afaa9fc768a14e72ab1ff46bfc813`
- runtime: Hugging Face Transformers in-process
- processor/model loading: `AutoProcessor` and `AutoModelForMultimodalLM`
- Transformers: `5.16.1`
- dtype: `bfloat16`
- generation: `do_sample=False`
- `max_new_tokens=8192`
- `repetition_penalty=1.05`
- registry capability: GPU only

Tabulus maps `--device gpu:0` to the PyTorch device `cuda:0`. The processor
and model are loaded lazily and reused across crops by the batch layer.

The implemented adapter path does not require Docker, vLLM, or
`qwen-vl-utils`.

## OTSL Normalization

TRivia produces native OTSL output. Tabulus preserves the raw OTSL in the
native reconstruction artifact, then performs deterministic OTSL-to-HTML
normalization through `tabulus.table_ocr.parsing.otsl_table_to_html`.

The supported structural tokens are:

- `fcel`
- `ecel`
- `lcel`
- `ucel`
- `xcel`
- `nl`

Ragged generated rows are rectangularized to the width of the widest OTSL row.
This preserves the model's generated structure for the shared HTML parser and
CSV-style output. It is not semantic correction, column-count guessing, or
heuristic repair of the model output.

## Native Output

The adapter preserves TRivia model metadata, revision, prompt, dtype,
generation settings, device/version metadata, image dimensions, token counts,
raw OTSL, and the Tabulus normalization function as native evidence:

```text
<crop-root>/
  reconstructions/
    trivia/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be `ok` and exactly one structured table must parse from the crop. Native
and parsed evidence is retained for empty or ambiguous results.

## Limitations

This adapter reconstructs one physical canonical MinerU crop at a time. It does
not independently locate or crop tables from the source PDF, semantically
correct cell contents, merge continued tables, classify reference tables,
extract bibliographies, match references, resolve DOI values, or write final
resolved CSV files.

The integration validation confirms adapter behavior and artifact generation.
It does not establish that TRivia is more accurate, better, or worse than any
other reconstruction candidate.
