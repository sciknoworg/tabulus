# Dolphin-v2

Dolphin-v2 is a vision-language model used by Tabulus for table
reconstruction from canonical MinerU table crops. The Tabulus adapter sends
the crop directly to the ByteDance Dolphin-v2 checkpoint and expects native
HTML table output.

## Official Resources

- [Dolphin-v2 model](https://huggingface.co/ByteDance/Dolphin-v2)

## Role In Tabulus

The registered Tabulus adapter is:

```text
dolphin-v2
```

The exact checkpoint used by Tabulus is `ByteDance/Dolphin-v2` at revision
`c37c62768c644bb594da4283149c627765aa80f3`. Its underlying backbone
architecture is Qwen2.5-VL, implemented through the Transformers class
`Qwen2_5_VLForConditionalGeneration`. Tabulus is not substituting a generic
Qwen checkpoint for Dolphin-v2.

Dolphin-v2 receives the same canonical MinerU crop used by the other
crop-consuming reconstruction adapters. It does not redetect tables, run
page-level layout detection, choose a different crop from the source PDF,
recrop based on model output, or perform margin cropping.

The reconstruction path is:

```text
canonical MinerU crop
        |
        v
Dolphin-v2
        |
        v
native HTML table output
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

Run Dolphin-v2 through the shared reconstruction CLI:

```bash
export CUDA_VISIBLE_DEVICES=0

tabulus reconstruct-tables \
  --crops-folder /path/to/tabulus-output/table-crops \
  --adapter dolphin-v2 \
  --device gpu:0
```

`CUDA_VISIBLE_DEVICES` can remap physical GPU numbering, so a selected
physical GPU may appear to the process as `cuda:0`.

## Settings Used By Tabulus

The validated integration uses:

- model checkpoint: `ByteDance/Dolphin-v2`
- model revision: `c37c62768c644bb594da4283149c627765aa80f3`
- backbone architecture: Qwen2.5-VL
- Transformers model class: `Qwen2_5_VLForConditionalGeneration`
- Transformers: `4.51.0`
- runtime: Hugging Face Transformers in-process
- model precision: `bfloat16`
- prompt: `Parse the table in the image.`
- `max_new_tokens=4096`
- `do_sample=False`
- `temperature=None`
- registry capability: GPU only

The deterministic generation settings differ from simply inheriting the
checkpoint's sampling configuration. Tabulus uses deterministic generation so
that repeated reconstruction of the same canonical crop with the same model
revision is reproducible for benchmarking. Tabulus does not perform semantic
correction to make outputs deterministic.

The validated software environment included Python 3.12, PyTorch 2.6.0,
torchvision 0.21.0, Transformers 4.51.0, Accelerate 1.4.0, and
`qwen-vl-utils` 0.0.14. These are validated environment details; consult the
implementation when deciding the exact dependencies for a runtime image.

## Image Preprocessing

The Dolphin element-processing path operates directly on the canonical crop.
The only image preprocessing applied by Tabulus is deterministic model-input
preparation:

- convert the crop image to RGB
- apply Dolphin's official `resize_img`-style resizing
- constrain the maximum side to 1600 pixels
- constrain the minimum side to 28 pixels

This resizing is not table redetection or recropping.

## Native Output

Dolphin-v2 produces native HTML table markup. Tabulus preserves the raw model
generation, also preserves a clean version with model special tokens removed
for parsing, and passes the clean HTML to:

```text
tabulus.table_ocr.parsing:parse_table_text
```

Native format is HTML and Dolphin-specific normalization is `none`. The shared
parser processes the model-native structure deterministically. Tabulus does
not correct OCR values, fix chemical names, modify references, repair
`rowspan` or `colspan` values based on expected table structure, or force the
output to resemble another reconstruction candidate.

Native Dolphin-v2 artifacts preserve model provenance and native evidence,
including model repository and revision, backbone architecture, model class,
prompt, requested and resolved dtype, generation settings, execution device,
Transformers/PyTorch/`qwen-vl-utils` versions, source and model-input image
sizes, token counts, raw output, clean output, native format, complete HTML
table count, canonical-crop input policy, image-preprocessing policy, and
explicit `layout_redetection=False` and `recropping=False` flags.

The adapter writes the standard reconstruction artifact layers:

```text
<crop-root>/
  reconstructions/
    dolphin-v2/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be `ok` and exactly one structured table must parse from the crop. Native
and parsed evidence is retained for empty or ambiguous results.

## Large Tables

Dolphin-v2 uses a 4096-token maximum generation length. If generation reaches
that limit before producing a complete HTML table, Tabulus does not invent the
missing remainder or repair the table. The result is retained as native
evidence, marked empty, and no prediction CSV is written.

This behavior keeps reconstruction outputs faithful to the candidate model
rather than semantically repairing them inside Tabulus.

## Validation Notes

The implementation has passed focused adapter tests, full-suite validation,
deterministic reproducibility checks, and real CLI reconstruction checks.
These are engineering validation results, not reconstruction accuracy
measurements or model-ranking evidence.

## Limitations

This adapter reconstructs one physical canonical MinerU crop at a time. It
does not independently locate or crop tables from the source PDF, run
page-level layout detection, semantically correct cell contents, merge
continued tables, classify reference tables, extract bibliographies, match
references, resolve DOI values, or write final resolved CSV files.
