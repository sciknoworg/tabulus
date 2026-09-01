# Nanonets-OCR-s

Nanonets-OCR-s is a vision-language table reconstruction candidate used by
Tabulus for Stage 2 reconstruction from canonical MinerU table crops. The
Tabulus adapter sends the crop directly to the pinned Nanonets-OCR-s model
checkpoint and expects native structured HTML table output.

## Official Resources

- [Nanonets-OCR-s model](https://huggingface.co/nanonets/Nanonets-OCR-s)

## Role In Tabulus

The registered Tabulus adapter is:

```text
nanonets-ocr-s
```

The exact model checkpoint used by Tabulus is `nanonets/Nanonets-OCR-s` at
revision `3baad182cc87c65a1861f0c30357d3467e978172`. Its underlying backbone
architecture is Qwen2.5-VL, implemented through the runtime Transformers class
`Qwen2_5_VLForConditionalGeneration`. Tabulus is not substituting a generic
Qwen checkpoint for Nanonets-OCR-s.

Nanonets-OCR-s receives the same canonical MinerU crop used by the other
crop-consuming reconstruction adapters. It does not run external layout
redetection, table redetection, external recropping, semantic repair, or
continued-table merging. Each physical crop remains independent.

The reconstruction path is:

```text
canonical MinerU crop
        |
        v
Nanonets-OCR-s
        |
        v
native structured HTML
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

Run Nanonets-OCR-s through the shared reconstruction CLI:

```bash
export CUDA_VISIBLE_DEVICES=0

python -m tabulus.cli reconstruct-tables \
  --crops <canonical-crop-directory> \
  --adapter nanonets-ocr-s \
  --device gpu:0
```

For multiple canonical crop roots:

```bash
python -m tabulus.cli reconstruct-tables \
  --crops-folder <table-crops-root> \
  --adapter nanonets-ocr-s \
  --device gpu:0
```

Nanonets-OCR-s operates on the canonical table-crop handoff produced earlier
in the Tabulus pipeline. It does not perform the initial PDF profiling or
table-cropping stage.

## Settings Used By Tabulus

The validated integration uses:

- model checkpoint: `nanonets/Nanonets-OCR-s`
- model revision: `3baad182cc87c65a1861f0c30357d3467e978172`
- backbone architecture: Qwen2.5-VL
- Transformers model class: `Qwen2_5_VLForConditionalGeneration`
- processor: `AutoProcessor`
- processor setting: `use_fast=False`
- model loader: `AutoModelForImageTextToText`
- Transformers: `4.52.4`
- tokenizers: `0.21.4`
- FlashAttention: `2.7.3`
- PyTorch: `2.6.0+cu124`
- attention implementation: `flash_attention_2`
- model dtype: `bfloat16`
- system message: `You are a helpful assistant.`
- `max_new_tokens=15000`
- `do_sample=False`
- registry capability: GPU only

The explicit `use_fast=False` processor setting is intentional. The first
successful feasibility run used the default slow image processor, and setting
`use_fast=False` reproduced that output bit-for-bit.

The validated environment name was `tabulus-nanonets-ocr-s` on an NVIDIA L40S.
The adapter also imports PyTorch, torchvision, Pillow, and Accelerate at
runtime. Those are validated environment and dependency details, not
reconstruction-accuracy claims.

The implemented adapter path does not require Docker, vLLM, or
`qwen-vl-utils`.

The prompt used by the adapter is:

```text
Extract the text from the above document as if you were reading it naturally.
Return the tables in html format. Return the equations in LaTeX representation.
If there is an image in the document and image caption is not present, add a
small description of the image inside the <img></img> tag; otherwise, add the
image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex:
<watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in
brackets.
Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer
using ☐ and ☑ for check boxes.
```

Tabulus explicitly runs deterministic generation with `do_sample=False`.
Transformers may warn that a temperature value inherited from the model
generation configuration is ignored; temperature is not an intentional Tabulus
generation parameter for this adapter.

## Input And Image Processing

The adapter converts the supplied canonical crop to RGB and passes it through
the model processor. Model-internal image preprocessing or resizing is allowed,
but it is not external Tabulus recropping or table redetection.

The common-crop policy is deliberate: reconstruction candidates are compared
on the same visual evidence rather than on candidate-specific detections or
crops from the original PDF.

## Native Output

The validated table test produced native structured HTML. Tabulus preserves
the raw decoded generation, creates the parser-facing clean generation by
removing model special tokens, and sends the clean model output directly to:

```text
tabulus.table_ocr.parsing:parse_table_text
```

There is no Nanonets-specific parser and no Nanonets-specific semantic or
structural normalization stage. Existing deterministic HTML parsing rules
apply unchanged.

The validated Nanonets output included rich HTML table constructs such as:

```text
<table>
<thead>
<tbody>
<th>
<td>
rowspan
colspan
<sup>
<sub>
<br>
```

Tabulus does not semantically fix or clean the model-generated table, remove
model-native blank rows, repair cell contents, merge continued tables, or
perform reference-resolution heuristics during reconstruction.

Native Nanonets-OCR-s artifacts preserve model provenance and native evidence,
including model repository and revision, backbone architecture, runtime model
class, processor/model loader settings, prompt and system message, generation
settings, dtype, attention implementation, execution device, dependency
versions, raw generation, clean parser-facing generation, detected table
counts, parser errors, and the canonical-crop input policy.

The adapter writes the standard reconstruction artifact layers:

```text
<crop-root>/
  reconstructions/
    nanonets-ocr-s/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be `ok` and exactly one structured table must parse from the crop. Native
and parsed evidence is retained for empty or ambiguous results.

## Validation Notes

Integration validation covered deterministic direct model execution, the
shared parser path, a real end-to-end CLI reconstruction, and the full
multi-crop engineering run recorded in {doc}`../project-notes/current-state`.
Focused Nanonets adapter tests reported 10 passed, and the complete Tabulus
test suite reported 216 passed after integration.

These are engineering and integration validation observations. They are not
gold-standard reconstruction accuracy, precision, recall, F1, runtime
guarantees, or evidence that Nanonets-OCR-s is better or worse than another
adapter.

## Limitations

This adapter reconstructs one physical canonical MinerU crop at a time. It
does not independently locate or crop tables from the source PDF, semantically
correct cell contents, merge continued tables, classify reference tables,
extract bibliographies, match references, resolve DOI values, or write final
resolved CSV files.
