# InternVL3.5-8B

InternVL3.5-8B is a general-purpose multimodal vision-language model used by
Tabulus for Stage 2 table reconstruction from canonical MinerU table crops.
Unlike the dedicated OCR/document/table-parsing models in the candidate set,
this adapter evaluates a broader VLM on the same fixed reconstruction input.

## Official Resources

- [InternVL project repository](https://github.com/OpenGVLab/InternVL)
- [InternVL3.5-8B-HF model used by Tabulus](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF)

## Role In Tabulus

The registered Tabulus adapter is:

```text
internvl3-5-8b
```

The exact model checkpoint used by Tabulus is `OpenGVLab/InternVL3_5-8B-HF` at
revision `741a7d03020411e666c6109218ab71e08151ef86`. The adapter verifies the
loaded model and processor classes:

- model class: `InternVLForConditionalGeneration`
- model type: `internvl`
- text model type: `qwen3`
- vision model type: `internvl_vision`
- processor class: `InternVLProcessor`
- image processor: `GotOcr2ImageProcessorFast`
- tokenizer: `Qwen2TokenizerFast`

InternVL3.5-8B receives the same canonical MinerU crop used by the other
crop-consuming reconstruction adapters. It does not process the original PDF,
run external layout redetection, run external table redetection, perform
bbox-based recropping, semantically repair table contents, resolve references,
or merge continued tables.

The adapter uses native Hugging Face Transformers loading with
`local_files_only=True`. The pinned model snapshot must already be present in
the local Hugging Face cache; the adapter does not perform runtime model
downloads.

The reconstruction path is:

```text
canonical MinerU crop
        |
        v
official InternVL image preprocessing / model-native tiling
        |
        v
InternVL generation
        |
        v
native HTML
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

Run InternVL3.5-8B through the shared reconstruction CLI:

```bash
export CUDA_VISIBLE_DEVICES=0

tabulus reconstruct-tables \
  --crops <canonical-crop-directory> \
  --adapter internvl3-5-8b \
  --device gpu:0
```

For multiple canonical crop roots:

```bash
tabulus reconstruct-tables \
  --crops-folder <table-crops-root> \
  --adapter internvl3-5-8b \
  --device gpu:0
```

InternVL3.5-8B operates on the canonical table-crop handoff produced earlier
in the Tabulus pipeline. It does not perform PDF profiling, table localization,
or canonical crop generation.

## Settings Used By Tabulus

The validated integration uses:

- model: `OpenGVLab/InternVL3_5-8B-HF`
- model revision: `741a7d03020411e666c6109218ab71e08151ef86`
- backend: native Hugging Face Transformers
- model loader: `InternVLForConditionalGeneration`
- processor: `AutoProcessor` / `InternVLProcessor`
- local model loading: `local_files_only=True`
- model dtype: `bfloat16`
- floating-point image tensors: `bfloat16`
- attention implementation: SDPA
- FlashAttention: not required
- task: `table_to_html`
- prompt source: Tabulus-defined
- `max_new_tokens=8192`
- `do_sample=False`
- `num_beams=1`
- `temperature=1.0`
- `top_p=1.0`
- `repetition_penalty=1.0`
- registry capability: GPU only

Do not enable `use_flash_attn=True` for this documented integration. The
frozen InternVL3.5-8B adapter configuration uses SDPA under the validated
Transformers version.

The fixed Tabulus prompt is:

```text
Extract the table from this image and output only a valid HTML table. Return only the table markup, with no explanation, no markdown fences, and no extra text before or after the table.
```

`max_new_tokens=8192` is a hard generation safety ceiling. It is not a typical
output length and is not required for valid tables. If generation reaches this
ceiling, the adapter treats the result as empty rather than accepting truncated
or degenerative output.

## Native Output

InternVL3.5-8B produces native HTML for this Tabulus task. The adapter does not
rewrite or repair that HTML before parsing. The generated output is passed to
the existing shared parser:

```text
tabulus.table_ocr.parsing:parse_table_text
```

There is no InternVL-specific structural or semantic normalization stage.
Tabulus does not correct cell contents, infer missing structure from domain
knowledge, merge continued tables, perform reference-resolution heuristics, or
clean up malformed HTML beyond the shared parser behavior.

The adapter writes the standard reconstruction artifact layers:

```text
<crop-root>/
  reconstructions/
    internvl3-5-8b/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be `ok` and exactly one structured table must parse from the crop. If
multiple parsed tables are returned, Tabulus preserves the native and parsed
evidence without arbitrarily selecting or merging one. Prediction CSVs are
pre-reference-resolution artifacts.

## Validated Configuration

The validated software environment used Python 3.12.14, PyTorch 2.7.0+cu128,
torchvision 0.22.0+cu128, Transformers 4.55.0, Accelerate 1.14.0, BF16 model
weights and floating-point image tensors, and SDPA attention. The current
Tabulus registry marks InternVL3.5-8B as GPU-only. These are implementation
and reproducibility details, not reconstruction-accuracy claims.

## Limitations

This adapter reconstructs one physical canonical MinerU crop at a time. It
does not independently locate or crop tables from the source PDF, run an
external layout detector, use bounding boxes for recropping, semantically
correct cell contents, merge continued tables, classify reference tables,
extract bibliographies, match references, resolve DOI values, or write final
resolved CSV files. Model-native image preprocessing and tiling are allowed,
but they are distinct from Tabulus externally redetecting or recropping a
table.
