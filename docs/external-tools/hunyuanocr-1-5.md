# HunyuanOCR-1.5

HunyuanOCR-1.5 is a document vision-language model used by Tabulus for Stage 2
table reconstruction from canonical MinerU table crops. The Tabulus adapter
uses the model's dedicated table task through direct Hugging Face Transformers
inference.

## Official Resources

- [HunyuanOCR project repository](https://github.com/Tencent-Hunyuan/HunyuanOCR)
- [HunyuanOCR model used by Tabulus](https://huggingface.co/tencent/HunyuanOCR)

## Role In Tabulus

The registered Tabulus adapter is:

```text
hunyuanocr-1-5
```

The exact model checkpoint used by Tabulus is `tencent/HunyuanOCR` at revision
`47644ecc4fc854efa4f505155158831f36773ee4`. The adapter verifies that the
loaded model class is `HunYuanVLForConditionalGeneration` and that the model
type is `hunyuan_vl`.

HunyuanOCR-1.5 receives the same canonical MinerU crop used by the other
crop-consuming reconstruction adapters. It does not process the original PDF,
run external layout redetection, run external table redetection, externally
recrop the image, semantically repair table contents, resolve references, or
merge continued tables.

The implemented Tabulus adapter uses direct Transformers inference. It does
not require vLLM, DFlash, Docker, or another model-serving process for this
adapter path.

The reconstruction path is:

```text
canonical MinerU crop
        |
        v
HunyuanOCR-1.5 official table task
        |
        v
native generated HTML
        |
        v
official HunyuanOCR repetition safeguards
        |
        v
shared Tabulus HTML parser
        |
        v
prediction CSV when exactly one table parses
```

For the generic adapter interface and artifact contract, see
{doc}`../modules/table-ocr-adapters`.

## Invocation

Run HunyuanOCR-1.5 through the shared reconstruction CLI:

```bash
export CUDA_VISIBLE_DEVICES=0

tabulus reconstruct-tables \
  --crops <canonical-crop-directory> \
  --adapter hunyuanocr-1-5 \
  --device gpu:0
```

For multiple canonical crop roots:

```bash
tabulus reconstruct-tables \
  --crops-folder <table-crops-root> \
  --adapter hunyuanocr-1-5 \
  --device gpu:0
```

HunyuanOCR-1.5 operates on the canonical table-crop handoff produced earlier
in the Tabulus pipeline. It does not perform PDF profiling, table localization,
or canonical crop generation.

## Settings Used By Tabulus

The validated integration uses:

- model: `tencent/HunyuanOCR`
- model revision: `47644ecc4fc854efa4f505155158831f36773ee4`
- model class: `HunYuanVLForConditionalGeneration`
- model type: `hunyuan_vl`
- runtime: direct Hugging Face Transformers inference
- processor: `AutoProcessor` with `use_fast=False`
- model dtype: `bfloat16`
- attention implementation: `eager`
- task: table
- prompt: `把图中的表格解析为HTML。`
- `max_new_tokens=8192`
- `do_sample=False`
- `repetition_penalty=1.08`
- `use_cache=True`
- registry capability: GPU only

The prompt is the official table prompt used by the implementation. In prose,
it instructs the model to parse the table in the image as HTML.

HunyuanOCR's inference-time repetition safeguards are preserved by the adapter.
Tail-repetition stopping uses:

- minimum repeats: 8
- maximum repeated unit length: 256 characters
- first decoded-character check: 4000 characters
- subsequent checks: every 1000 decoded characters
- inspected tail window: 8000 characters
- token probe interval: 64 tokens

Final repeated-suffix cleanup uses a minimum repeat count of 10. These are
model inference safeguards against autoregressive repetition degeneration, not
Tabulus semantic table repair. The native artifact records whether tail
stopping triggered and whether repeated-suffix cleanup changed the output.

## Native Output

HunyuanOCR-1.5 produces native HTML for the table task. Tabulus preserves
separate representations for provenance:

- `raw_output`: decoded model generation with special tokens retained
- `decoded_output`: generation with model special tokens removed
- `clean_output`: parser-facing output after the official repeated-suffix cleanup

No document-level Markdown normalization is applied to this table task. The
clean HTML is passed to the existing shared parser:

```text
tabulus.table_ocr.parsing:parse_table_text
```

There is no HunyuanOCR-specific structural or semantic normalization stage.
If multiple HTML tables or structured tables are produced, Tabulus preserves
them independently and does not arbitrarily select, collapse, concatenate, or
merge them.

The adapter writes the standard reconstruction artifact layers:

```text
<crop-root>/
  reconstructions/
    hunyuanocr-1-5/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be `ok` and exactly one structured table must parse from the crop.

## Validated Configuration

The validated software environment used Python 3.12, Transformers 5.13.0,
Accelerate 1.14.0, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, bfloat16,
and eager attention. The current Tabulus registry marks HunyuanOCR-1.5 as
GPU-only. These are implementation and reproducibility details, not
reconstruction-accuracy claims.

## Limitations

This adapter reconstructs one physical canonical MinerU crop at a time. It
does not independently locate or crop tables from the source PDF, run
page-level layout detection, semantically correct cell contents, merge
continued tables, classify reference tables, extract bibliographies, match
references, resolve DOI values, or write final resolved CSV files. Model- or
processor-internal image preprocessing is allowed, but it is distinct from
Tabulus externally redetecting or recropping a table.
