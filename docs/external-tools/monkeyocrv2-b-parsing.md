# MonkeyOCRv2-B-Parsing

MonkeyOCRv2-B-Parsing is a vision-language table reconstruction candidate used
by Tabulus for Stage 2 reconstruction from canonical MinerU table crops. The
Tabulus adapter uses MonkeyOCRv2's direct table-recognition task rather than
the full document-layout pipeline.

## Official Resources

- [MonkeyOCRv2-B-Parsing model](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing)

## Role In Tabulus

The registered Tabulus adapter is:

```text
monkeyocrv2-b-parsing
```

The exact model checkpoint used by Tabulus is
`zenosai/MonkeyOCRv2-B-Parsing` at revision
`2419139b7bcd3fda2689b2a83167172afba91c8b`.

MonkeyOCRv2-B-Parsing receives the same canonical MinerU crop used by the
other crop-consuming reconstruction adapters. It does not run external layout
redetection, external table redetection, external recropping, semantic repair,
or continued-table merging. Each physical crop remains independent.

The reconstruction path is:

```text
canonical MinerU crop
        |
        v
MonkeyOCRv2-B-Parsing direct table recognition
        |
        v
native OTSL
        |
        v
Tabulus OTSL-to-HTML normalization
        |
        v
shared span-aware HTML parser
        |
        v
prediction CSV when exactly one table parses
```

For the generic adapter interface and artifact contract, see
{doc}`../modules/table-ocr-adapters`.

## Invocation

Run MonkeyOCRv2-B-Parsing through the shared reconstruction CLI:

```bash
export CUDA_VISIBLE_DEVICES=0

tabulus reconstruct-tables \
  --crops <canonical-crop-directory> \
  --adapter monkeyocrv2-b-parsing \
  --device gpu:0
```

For multiple canonical crop roots:

```bash
tabulus reconstruct-tables \
  --crops-folder <table-crops-root> \
  --adapter monkeyocrv2-b-parsing \
  --device gpu:0
```

MonkeyOCRv2-B-Parsing operates on the canonical table-crop handoff produced
earlier in the Tabulus pipeline. It does not perform PDF profiling, table
localization, or canonical crop generation.

## Settings Used By Tabulus

The validated integration uses:

- model checkpoint: `zenosai/MonkeyOCRv2-B-Parsing`
- model revision: `2419139b7bcd3fda2689b2a83167172afba91c8b`
- runtime: direct Hugging Face Transformers inference
- processor: `AutoProcessor`
- processor setting: `use_fast=False`
- model loader: `AutoModelForCausalLM`
- resolved model class: `MonkeyOCRv2ForCausalLM`
- custom Hugging Face model code: `trust_remote_code=True`
- attention implementation: SDPA
- FlashAttention: not required
- model dtype: `bfloat16`
- task: direct single-task table recognition
- prompt: `Please extract the table from the image and represent it in OTSL format.`
- `max_new_tokens=4096`
- `do_sample=False`
- `temperature=None`
- `top_p=None`
- `top_k=None`
- processor table `min_pixels=1003520`
- native model output: OTSL
- registry capability: GPU only

The validated software environment used Python 3.11, Transformers 4.57.1,
Accelerate 1.11.0, timm 1.0.27, einops 0.8.1, PyTorch 2.6.0+cu124, and
torchvision 0.21.0+cu124. The adapter also requires Pillow for image loading.
These are validated environment details, not reconstruction-accuracy claims.

The implemented adapter path does not use vLLM or DFlash.

## Input And Image Processing

The adapter sends the supplied canonical crop to MonkeyOCRv2-B-Parsing's model
processor. Model- or processor-internal resizing is allowed, but it is not
external Tabulus recropping or table redetection.

The common-crop policy is deliberate: reconstruction candidates are compared
on the same visual evidence rather than on candidate-specific detections or
crops from the original PDF.

## Native Output

MonkeyOCRv2-B-Parsing produces native OTSL. Tabulus preserves the raw model
generation, removes model special tokens only for the parser-facing
representation, converts OTSL deterministically through:

```text
tabulus.table_ocr.parsing:otsl_table_to_html
```

and then sends the resulting HTML to the existing shared table parser:

```text
tabulus.table_ocr.parsing:parse_table_text
```

There is no MonkeyOCRv2-specific semantic repair stage. Tabulus does not
correct cell contents, infer missing structure from domain knowledge, merge
continued tables, or perform reference-resolution heuristics during
reconstruction.

The adapter writes the standard reconstruction artifact layers:

```text
<crop-root>/
  reconstructions/
    monkeyocrv2-b-parsing/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be `ok` and exactly one structured table must parse from the crop. Native
and parsed evidence is retained for empty or ambiguous results.

## Limitations

This adapter reconstructs one physical canonical MinerU crop at a time. It
does not independently locate or crop tables from the source PDF, run
MonkeyOCRv2's full document-layout pipeline, semantically correct cell
contents, merge continued tables, classify reference tables, extract
bibliographies, match references, resolve DOI values, or write final resolved
CSV files.
