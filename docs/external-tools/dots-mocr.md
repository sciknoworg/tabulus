# dots.mocr

dots.mocr is a document vision-language model used by Tabulus for Stage 2
table reconstruction from canonical MinerU table crops. The Tabulus adapter
uses the active dots.mocr layout prompt through direct Hugging Face
Transformers inference.

## Official Resources

- [dots.mocr project repository](https://github.com/studio-dots-ai/dots.mocr)
- [dots.mocr model used by Tabulus](https://huggingface.co/dots-studio/dots.mocr)

## Role In Tabulus

The registered Tabulus adapter is:

```text
dots-mocr
```

The exact model checkpoint used by Tabulus is `dots-studio/dots.mocr` at
revision `e539fbb52280393adc081b289ec597430a0f9031`. The adapter verifies the
pinned remote-code configuration and model classes:

- config class: `DotsOCRConfig`
- model class: `DotsOCRForCausalLM`
- model type: `dots_ocr`
- processor class: `DotsVLProcessor`
- image processor: `Qwen2VLImageProcessorFast`
- tokenizer: `Qwen2TokenizerFast`

dots.mocr receives the same canonical MinerU crop used by the other
crop-consuming reconstruction adapters. It does not process the original PDF,
run external layout redetection, run external table redetection, externally
recrop the image, perform semantic repair, repair JSON, resolve references, or
merge continued tables.

The adapter uses direct Transformers inference. It does not use a vLLM server,
DFlash, or Docker for this Tabulus adapter path.

The reconstruction path is:

```text
canonical MinerU crop
        |
        v
dots.mocr active layout prompt (prompt_layout_all_en)
        |
        v
model-native JSON layout output
        |
        v
select model-emitted objects whose category == "Table"
        |
        v
preserve their model-emitted HTML
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

Run dots.mocr through the shared reconstruction CLI:

```bash
export CUDA_VISIBLE_DEVICES=0

tabulus reconstruct-tables \
  --crops <canonical-crop-directory> \
  --adapter dots-mocr \
  --device gpu:0
```

For multiple canonical crop roots:

```bash
tabulus reconstruct-tables \
  --crops-folder <table-crops-root> \
  --adapter dots-mocr \
  --device gpu:0
```

dots.mocr operates on the canonical table-crop handoff produced earlier in the
Tabulus pipeline. It does not perform PDF profiling, table localization, or
canonical crop generation.

## Settings Used By Tabulus

The validated integration uses:

- model: `dots-studio/dots.mocr`
- model revision: `e539fbb52280393adc081b289ec597430a0f9031`
- runtime: direct Hugging Face Transformers inference
- custom model code: `trust_remote_code=True`
- processor: pinned custom `DotsVLProcessor` with `use_fast=True`
- image processor: `Qwen2VLImageProcessorFast`
- tokenizer: `Qwen2TokenizerFast`
- model dtype: `bfloat16`
- attention implementation: `flash_attention_2`
- prompt mode: `prompt_layout_all_en`
- `max_new_tokens=24000`
- `do_sample=False`
- `num_beams=1`
- registry capability: GPU only

The active prompt asks dots.mocr to return document layout elements as JSON
with each element's bounding box, category, and text content. For `Table`
elements, the prompt asks for HTML in the `text` field:

```text
Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
```

Tabulus deliberately uses this active layout prompt path rather than a
table-only prompt.

The implementation verifies that the resolved generation defaults are
compatible with deterministic greedy generation. Temperature and top-p values
from the loaded model generation configuration are recorded as resolved model
metadata rather than introduced as custom sampling settings.

## Bounding-Box Policy

dots.mocr performs model-native layout detection inside the supplied canonical
crop and can emit bounding boxes for detected layout elements. Tabulus retains
those bounding boxes for provenance only.

They are not a second crop-generation stage, and Tabulus does not use them to
recrop the canonical image. The native artifact records:

- `bbox_policy="provenance_only"`
- `table_bboxes_used_for_recropping=False`

The scientific comparison policy remains that every crop-consuming
reconstruction candidate receives exactly the same canonical MinerU crop.

## Native Output

dots.mocr produces model-native JSON layout output with HTML inside
model-emitted `Table` objects. The native format is:

```text
json_layout_with_html_tables
```

Tabulus preserves the raw model generation. Model special tokens are removed
only to obtain the parser-facing JSON string. No semantic normalization or JSON
repair is applied.

The model output may be represented as a JSON list or as a JSON object in
practice. The adapter traverses the parsed native JSON structure to collect
layout objects without altering their contents. Only objects whose
`category == "Table"` are treated as table candidates, and each Table object's
HTML is preserved independently.

Table HTML is passed to the existing shared parser:

```text
tabulus.table_ocr.parsing:parse_table_text
```

If multiple structured tables result from one canonical crop, Tabulus
preserves them and does not arbitrarily select, concatenate, collapse, or merge
them.

The adapter writes the standard reconstruction artifact layers:

```text
<crop-root>/
  reconstructions/
    dots-mocr/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be `ok` and exactly one structured table must parse from the crop.
Prediction CSVs are pre-reference-resolution artifacts.

## Empty Results

The adapter reports an explicit empty result when dots.mocr returns no
generated layout output, returns invalid JSON layout output, emits no layout
objects, emits no model-native `Table` objects, or emits Table objects without
usable HTML tables. These cases preserve native evidence where available and
are not treated as parser repair opportunities.

## Validated Configuration

The validated software environment used Python 3.12, Transformers 4.57.6,
Accelerate 1.14.0, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128,
qwen-vl-utils 0.0.14, FlashAttention 2.8.0.post2, bfloat16, and
`flash_attention_2`. The current Tabulus registry marks dots.mocr as GPU-only.
These are implementation and reproducibility details, not
reconstruction-accuracy claims.

## Limitations

This adapter reconstructs one physical canonical MinerU crop at a time. It
does not independently locate or crop tables from the source PDF, run an
external crop-generation stage, semantically correct cell contents, repair
malformed JSON, merge continued tables, classify reference tables, extract
bibliographies, match references, resolve DOI values, or write final resolved
CSV files. Model-native multimodal image processing is allowed, but it is
distinct from Tabulus externally redetecting or recropping a table.
