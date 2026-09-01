# NVIDIA Nemotron Parse v1.2

NVIDIA Nemotron Parse v1.2 is a document vision-language model used by
Tabulus for Stage 2 table reconstruction from canonical MinerU table crops.
The Tabulus adapter uses the model directly through Hugging Face Transformers
and does not run a separate OCR engine or a page-layout pipeline.

## Official Resources

- [NVIDIA Nemotron Parse v1.2 model](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)
- [C-RADIOv2-H model dependency](https://huggingface.co/nvidia/C-RADIOv2-H)

## Role In Tabulus

The registered Tabulus adapter is:

```text
nemotron-parse-v1-2
```

The exact model checkpoint used by Tabulus is
`nvidia/NVIDIA-Nemotron-Parse-v1.2` at revision
`2bd0189bffd6cdded6280d9f22a4077b25a504e3`. The model uses the transitive
`nvidia/C-RADIOv2-H` implementation; Tabulus verifies at runtime that the
loaded C-RADIO code resolves to revision
`0d8f4c18c877166eda07ddae1386bcad256b7a6a`.

NVIDIA Nemotron Parse v1.2 receives the same canonical MinerU crop used by the
other crop-consuming reconstruction adapters. It does not run external layout
redetection, external table redetection, external recropping, semantic repair,
reference resolution, or continued-table merging. Generated bounding boxes are
preserved as provenance only and are not used to recrop the image.

The reconstruction path is:

```text
canonical MinerU crop
        |
        v
NVIDIA Nemotron Parse v1.2
        |
        v
grounded semantic objects
        |
        v
Table-class LaTeX/tabular content
        |
        v
pinned NVIDIA table postprocessing to HTML
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

Run NVIDIA Nemotron Parse v1.2 through the shared reconstruction CLI:

```bash
export CUDA_VISIBLE_DEVICES=0

tabulus reconstruct-tables \
  --crops <canonical-crop-directory> \
  --adapter nemotron-parse-v1-2 \
  --device gpu:0
```

For multiple canonical crop roots:

```bash
tabulus reconstruct-tables \
  --crops-folder <table-crops-root> \
  --adapter nemotron-parse-v1-2 \
  --device gpu:0
```

NVIDIA Nemotron Parse v1.2 operates on the canonical table-crop handoff
produced earlier in the Tabulus pipeline. It does not perform PDF profiling,
table localization, or canonical crop generation.

## Settings Used By Tabulus

The validated integration uses:

- model: `nvidia/NVIDIA-Nemotron-Parse-v1.2`
- model revision: `2bd0189bffd6cdded6280d9f22a4077b25a504e3`
- resolved model class: `NemotronParseForConditionalGeneration`
- C-RADIO dependency: `nvidia/C-RADIOv2-H`
- expected C-RADIO revision: `0d8f4c18c877166eda07ddae1386bcad256b7a6a`
- runtime: direct Hugging Face Transformers inference
- model dtype: `bfloat16`
- attention implementation: SDPA
- model image canvas: `[2048, 1664]`
- prompt: `</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>`
- `max_new_tokens=9000`
- `do_sample=False`
- `num_beams=1`
- `repetition_penalty=1.1`
- registry capability: GPU only

The adapter uses NVIDIA generation-time processors from the pinned Nemotron
helper code:

- `TableInsertionLogitsProcessor` with table prefix `\begin{tabular}`
- `RepetitionStopProcessor` with `max_repetitions=10`,
  `ngram_sizes=[3, 4, 5, 6]`, and `window_size=500`

The helper files `hf_logits_processor.py`, `postprocessing.py`, and
`latex2html.py` are loaded from the pinned Nemotron model revision and must
already be available in the local Hugging Face cache. The adapter does not
silently fetch those helper files during inference.

## Native Output

NVIDIA Nemotron Parse v1.2 produces grounded semantic objects containing class,
bounding-box, and text information. Table-class object text is represented
natively as LaTeX/tabular content. Tabulus preserves the generated objects,
bounding boxes, raw output, clean output, model and C-RADIO revisions,
runtime versions, prompt, generation settings, source image size, and parser
provenance as native evidence.

Tabulus converts Table-class output to HTML with NVIDIA's pinned
`postprocess_text(table_format='HTML')` helper and then sends the HTML to the
existing shared parser:

```text
tabulus.table_ocr.parsing:parse_table_text
```

There is no Nemotron-specific semantic repair stage. Tabulus does not correct
cell contents, interpret bounding boxes as a new crop, infer missing structure
from domain knowledge, merge continued tables, or perform reference-resolution
heuristics during reconstruction.

The adapter writes the standard reconstruction artifact layers:

```text
<crop-root>/
  reconstructions/
    nemotron-parse-v1-2/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be `ok` and exactly one structured table must parse from the crop. If
multiple Table objects or multiple structured tables are returned, Tabulus
preserves the native and parsed evidence without arbitrarily choosing or
merging one.

## Validated Configuration

The validated software environment used Python 3.12, Transformers 5.6.1,
Accelerate 1.12.0, albumentations 2.0.8, timm 1.0.22, einops 0.8.2,
open-clip-torch 3.3.0, opencv-python-headless 5.0.0.93, beautifulsoup4 4.15.0,
PyTorch 2.6.0+cu124, and torchvision 0.21.0+cu124. These are implementation
and reproducibility details, not reconstruction-accuracy claims.

## Limitations

This adapter reconstructs one physical canonical MinerU crop at a time. It
does not independently locate or crop tables from the source PDF, run a
separate OCR engine, run page-level layout detection, semantically correct cell
contents, merge continued tables, classify reference tables, extract
bibliographies, match references, resolve DOI values, or write final resolved
CSV files.
