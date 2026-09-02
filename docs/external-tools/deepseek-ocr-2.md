# DeepSeek-OCR-2

DeepSeek-OCR-2 is a vision-language table reconstruction candidate used by
Tabulus for Stage 2 reconstruction from canonical MinerU table crops. The
Tabulus adapter sends the crop directly to the pinned DeepSeek-OCR-2 model
revision and passes the returned model output unchanged to the shared parser.

## Official Resources

- [DeepSeek-OCR-2 model](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)

## Role In Tabulus

The registered Tabulus adapter is:

```text
deepseek-ocr-2
```

The exact model repository is `deepseek-ai/DeepSeek-OCR-2` at revision
`aaa02f3811945a91062062994c5c4a3f4c0af2b0`. The resolved model class in the
validated configuration is `DeepseekOCR2ForCausalLM`.

DeepSeek-OCR-2 receives the same canonical MinerU crop used by the other
crop-consuming reconstruction adapters. It does not redetect the table from
the original PDF, run another layout detector over the PDF, choose a different
physical table region, externally recrop the canonical crop, expand the crop
using neighboring PDF content, or merge continued tables across pages.

The reconstruction path is:

```text
canonical MinerU crop
        |
        v
DeepSeek-OCR-2 model.infer(...)
        |
        v
model output with grounding/structured table content
        |
        v
shared HTML/Markdown parser
        |
        v
prediction CSV when exactly one table parses
```

For the generic adapter interface and artifact contract, see
{doc}`../modules/table-ocr-adapters`.

## Invocation

Run DeepSeek-OCR-2 through the shared reconstruction CLI:

```bash
export CUDA_VISIBLE_DEVICES=0

tabulus reconstruct-tables \
  --crops /path/to/tabulus-output/table-crops/<paper> \
  --adapter deepseek-ocr-2 \
  --device gpu:0
```

For multiple canonical crop roots:

```bash
tabulus reconstruct-tables \
  --crops-folder /path/to/tabulus-output/table-crops \
  --adapter deepseek-ocr-2 \
  --device gpu:0
```

Reference-table classification remains a downstream stage:

```bash
tabulus classify-reference-tables \
  --crops-folder /path/to/tabulus-output/table-crops \
  --adapter deepseek-ocr-2
```

## Settings Used By Tabulus

The validated integration uses:

- model repository: `deepseek-ai/DeepSeek-OCR-2`
- model revision: `aaa02f3811945a91062062994c5c4a3f4c0af2b0`
- resolved model class: `DeepseekOCR2ForCausalLM`
- runtime: Hugging Face Transformers with model-specific `infer(...)`
- `trust_remote_code=True`
- `use_safetensors=True`
- attention implementation: `flash_attention_2`
- model dtype: `bfloat16`
- prompt: `<image>\n<|grounding|>Convert the document to markdown.`
- `base_size=1024`
- `image_size=768`
- `crop_mode=True`
- `save_results=False`
- `eval_mode=True`
- registry capability: GPU only

The model and tokenizer are loaded from the pinned Hugging Face model
revision. The adapter uses DeepSeek-OCR-2's custom Transformers model code
from that revision; it is not described as downloading arbitrary GitHub source
code at runtime.

The adapter explicitly validates:

- `transformers==4.46.3`
- `tokenizers==0.20.3`
- `flash-attn==2.7.3`

The validated environment also included Python 3.12, PyTorch 2.6.0,
torchvision 0.21.0, Pillow, `einops`, `addict`, and `easydict` on an NVIDIA
L40S. Those are validated environment details; only the three versions above
are explicitly checked by the adapter.

The underlying DeepSeek eval-mode generation path uses `max_new_tokens=8192`,
`do_sample=False`, `temperature=0.0`, `no_repeat_ngram_size=35`, and
`use_cache=True`. `temperature=0.0` is present in the generation
configuration, but sampling is disabled; this should not be described as
stochastic temperature-based sampling.

## Model-Internal Dynamic Resolution

DeepSeek-OCR-2 uses model-internal visual preprocessing with:

```text
base_size: 1024
image_size: 768
crop_mode: True
model_internal_tiling: True
```

The parameter name `crop_mode` belongs to DeepSeek-OCR-2. In Tabulus this does
not mean external table redetection or a change to the canonical table crop.
It refers to DeepSeek-OCR-2 internally tiling/resizing the already supplied
canonical image for visual encoding.

## Native Output

DeepSeek-OCR-2 can emit grounding metadata followed by structured table
markup. A representative response can begin like:

```text
<|ref|>table<|/ref|><|det|>...</|det|>
<table>...</table>
```

Tabulus preserves the returned model output unchanged. The adapter records
`normalization: none` and `parser_input: model_infer_output_unchanged`.
There is no DeepSeek-specific semantic cleanup stage.

Do not assume DeepSeek-OCR-2 always emits HTML. Depending on the model
response, the shared parser can also accept structured Markdown table output.
The validated smoke-test output contained DeepSeek grounding metadata plus one
HTML table, and the shared parser extracted the table without DeepSeek-specific
preprocessing.

The parser is:

```text
tabulus.table_ocr.parsing:parse_table_text
```

Tabulus does not remove or semantically interpret grounding coordinates,
correct OCR text, fix chemical terminology, repair model-generated
`rowspan`/`colspan` values based on domain knowledge, compare the table
against another candidate, or perform semantic reconstruction repair.

Native DeepSeek-OCR-2 artifacts preserve model provenance and native evidence,
including model repository and revision, resolved model class, prompt,
requested and resolved dtype, attention implementation, execution device,
Transformers/tokenizers/FlashAttention/PyTorch/torchvision/Pillow versions,
source image size, decoded-output token count, output character count, raw and
clean model output, native-format description, normalization policy,
parser-input policy, detected HTML and structured table counts, parser errors,
canonical-input policy, dynamic-resolution settings, model-internal tiling,
layout-redetection and recropping flags, and `trust_remote_code` usage.

The adapter writes the standard reconstruction artifact layers:

```text
<crop-root>/
  reconstructions/
    deepseek-ocr-2/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be suitable for standard prediction export, which means status `ok` and
exactly one structured table parsed from the crop. Empty output and
multiple-table ambiguity preserve native and parsed evidence without silently
collapsing results into one CSV.

## Validation Notes

A real Tabulus CLI smoke test, reproducibility checks, focused adapter tests,
and full-suite validation have completed for this adapter. Some model-native
empty outputs have also been observed when DeepSeek-OCR-2 returned grounding
or text-like content without a structured table representation; Tabulus
retained those as `status="empty"` and did not write prediction CSVs.

These observations describe integration behavior only. They are not
reconstruction accuracy, precision, recall, F1, or evidence that
DeepSeek-OCR-2 is better or worse than another reconstruction backend.

## Limitations

This adapter reconstructs one physical canonical MinerU crop at a time. It
does not independently locate or crop tables from the source PDF, run an
external layout detector over the source PDF, semantically correct cell
contents, merge continued tables, classify reference tables, extract
bibliographies, match references, resolve DOI values, or write final resolved
CSV files.
