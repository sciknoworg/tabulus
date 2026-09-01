# Step 2: Table Reconstruction

## Goal

Turn canonical MinerU table crops into structured table reconstructions.

This stage is model-independent. A reconstruction adapter may use OCR, a
document vision-language model, or another table-reconstruction architecture,
but every crop-consuming adapter receives the same Tabulus handoff from Step 1.

## Input

The input is one canonical table-crop root:

```text
<crop-root>/
  tables_index.json
  images/
```

`tables_index.json` identifies physical MinerU-detected tables and records
their crop image paths, page numbers, bounding boxes, captions, footnotes,
MinerU provenance, and MinerU `table_body` values where available.

Every reconstruction candidate consumes the same canonical MinerU crop. The
comparison is:

```text
canonical MinerU crop
      |
      +--> reconstruction adapter A
      +--> reconstruction adapter B
      +--> reconstruction adapter C
      |
      v
structured reconstruction output
```

It is not:

```text
original PDF
      |
      v
each reconstruction model independently detects and crops tables
```

This boundary isolates table-reconstruction differences from table detection
and cropping differences. Each physical crop is processed independently; no
continued-table merging happens during reconstruction.

## Implemented Candidates

MinerU `table_body` is retained as a native reconstruction candidate produced
during PDF profiling. It is not a crop-consuming `tabulus.table_ocr` adapter.

The currently registered crop-consuming reconstruction adapters are:

| Adapter | Status | Details |
| --- | --- | --- |
| `paddleocr-vl` | Implemented | {doc}`../external-tools/paddleocr-vl` |
| `chandra` | Implemented | {doc}`../external-tools/chandra` |
| `nuextract3` | Implemented | {doc}`../external-tools/nuextract3` |
| `tesseract-tatr` | Implemented | {doc}`../external-tools/tesseract-tatr` |
| `rapidocr-tableformer` | Implemented | {doc}`../external-tools/docling` |
| `granite-vision-table` | Implemented | {doc}`../external-tools/granite-vision` |
| `trivia` | Implemented | {doc}`../external-tools/trivia` |
| `glm-ocr` | Implemented | {doc}`../external-tools/glm-ocr` |
| `dolphin-v2` | Implemented | {doc}`../external-tools/dolphin-v2` |
| `deepseek-ocr-2` | Implemented | {doc}`../external-tools/deepseek-ocr-2` |
| `nanonets-ocr-s` | Implemented | {doc}`../external-tools/nanonets-ocr-s` |

For the adapter interface and batch architecture, see
{doc}`../modules/table-ocr-adapters`.

## Reconstruction Flow

The common reconstruction path is:

```text
canonical MinerU crop
      |
      v
reconstruction adapter
      |
      v
adapter-native result
      |
      v
shared structural parsing
      |
      v
parsed table evidence
      |
      v
prediction CSV
      |
      v
reference-table classification
```

Adapter-native output is preserved before Tabulus parses it into the common
rectangular representation. Current adapters emit HTML, Markdown, JSON, or
model-specific evidence, but downstream Tabulus code receives the shared parsed
structure.

## CLI

Reconstruct one paper's canonical crops:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter paddleocr-vl \
  --device gpu:0
```

Choose another registered adapter with the same command shape:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter chandra \
  --device gpu:0
```

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter nuextract3 \
  --device gpu:0
```

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter tesseract-tatr \
  --device gpu:0
```

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter rapidocr-tableformer \
  --device gpu:0
```

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter granite-vision-table \
  --device gpu:0
```

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter trivia \
  --device gpu:0
```

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter glm-ocr \
  --device gpu:0
```

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter dolphin-v2 \
  --device gpu:0
```

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter deepseek-ocr-2 \
  --device gpu:0
```

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter nanonets-ocr-s \
  --device gpu:0
```

Granite Vision is an end-to-end vision-language-model reconstruction route,
not a conventional OCR adapter. The canonical MinerU crop is sent directly
to Granite Vision 4.1 4B, which generates OTSL from the `<tables_otsl>` prompt.
Docling parses that OTSL into structured cells, which Tabulus renders through
the shared reconstruction parser and output contract:

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
Docling OTSL parsing
      |
      v
structured cells
      |
      v
shared Tabulus parser/output contract
```

This adapter does not run Docling PDF conversion or page-layout/table
detection, and it does not redetect or recrop the original PDF. Each physical
MinerU crop remains independent.

TRivia-3B is also an end-to-end vision-language-model reconstruction route.
It receives the canonical MinerU crop directly, generates native OTSL, and
uses Tabulus-owned deterministic OTSL-to-HTML normalization before the shared
HTML table parser:

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
shared Tabulus parser/output contract
```

This normalization handles OTSL structure only. It does not semantically
correct cell contents, repair the model's table intent, merge continued tables,
or perform reference-resolution heuristics.

GLM-OCR is a GPU-only vision-language-model reconstruction route. It receives
the canonical MinerU crop directly and generates native HTML table output.
Tabulus preserves the raw generated output, removes model special tokens only
for the clean representation used for parsing, and passes the clean HTML to the
shared span-aware HTML parser:

```text
canonical MinerU crop
      |
      v
GLM-OCR
      |
      v
native HTML
      |
      v
shared Tabulus HTML parser
      |
      v
shared Tabulus parser/output contract
```

This adapter does not use the GLM-OCR SDK document pipeline, PP-DocLayout-V3,
layout redetection, candidate-specific recropping, or semantic repair of
model-generated HTML.

Dolphin-v2 is a GPU-only vision-language-model reconstruction route. Tabulus
uses the `ByteDance/Dolphin-v2` checkpoint, whose underlying backbone
architecture is Qwen2.5-VL, rather than substituting a generic Qwen checkpoint.
It receives the canonical MinerU crop directly, applies deterministic
Dolphin-style image preprocessing, and generates native HTML table markup:

```text
canonical MinerU crop
      |
      v
Dolphin-v2
      |
      v
native HTML
      |
      v
shared Tabulus HTML parser
      |
      v
shared Tabulus parser/output contract
```

The image preprocessing is RGB conversion plus Dolphin's official
`resize_img`-style resizing with maximum side 1600 pixels and minimum side 28
pixels. This is model-input preparation, not table redetection or recropping.
Dolphin-v2 uses deterministic generation in Tabulus (`do_sample=False`,
`temperature=None`) so repeated reconstruction of the same crop with the same
model revision is reproducible for benchmarking.

This adapter does not run page-level layout detection, choose a different crop
from the source PDF, perform margin cropping, semantically repair incomplete
HTML, correct cell contents, merge continued tables, or perform reference
resolution. If Dolphin-v2 reaches its 4096-token generation ceiling before a
complete HTML table is produced, Tabulus preserves the native evidence, marks
the result empty, and does not write a prediction CSV.

DeepSeek-OCR-2 is a GPU-only vision-language-model reconstruction route. It
receives the canonical MinerU crop directly and calls DeepSeek-OCR-2's
model-specific `infer(...)` method. DeepSeek can return grounding metadata
with structured HTML or Markdown table content; Tabulus passes that model
output unchanged to the shared parser:

```text
canonical MinerU crop
      |
      v
DeepSeek-OCR-2
      |
      v
grounding metadata plus structured table output
      |
      v
shared Tabulus HTML/Markdown parser
      |
      v
shared Tabulus parser/output contract
```

The DeepSeek parameter `crop_mode=True` refers to model-internal dynamic
resolution and tiling of the already supplied canonical image. It does not mean
that Tabulus externally redetects, expands, or recrops the table. This adapter
does not semantically interpret grounding coordinates, correct cell contents,
repair table structure, merge continued tables, or perform reference
resolution.

Nanonets-OCR-s is a GPU-only vision-language-model reconstruction route. It
uses the `nanonets/Nanonets-OCR-s` checkpoint, whose underlying backbone
architecture is Qwen2.5-VL, rather than treating Qwen2.5-VL as the adapter
identity. Nanonets receives the canonical MinerU crop directly and produces
native structured HTML:

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
shared Tabulus HTML parser
      |
      v
shared Tabulus parser/output contract
```

Tabulus preserves the raw decoded generation, removes model special tokens only
for the clean parser-facing representation, and passes the clean HTML through
the existing shared parser. There is no Nanonets-specific table parser,
semantic cell repair, external redetection, external recropping,
continued-table merging, or reference resolution during reconstruction.

For multiple papers, process every immediate child directory that contains a
`tables_index.json` file:

```bash
tabulus reconstruct-tables \
  --crops-folder "/path/to/tabulus-output/table-crops" \
  --adapter deepseek-ocr-2 \
  --device gpu:0
```

The crop input modes are mutually exclusive:

- `--crops <crop-root>`: process one canonical table-crop directory
- `--crops-folder <folder>`: process immediate child directories containing
  `tables_index.json`
- `--crops-list <text-file>`: process crop roots listed in a UTF-8 text file

For `--crops-folder`, discovery is non-recursive and crop roots are sorted
deterministically by directory name. For `--crops-list`, blank lines and lines
starting with `#` are ignored, relative paths are resolved relative to the list
file, and duplicate crop roots are rejected.

For multiple crop roots, Tabulus creates the selected adapter once, reuses that
instance across the complete batch, processes papers sequentially, preserves
independent per-paper outputs, and reports both per-paper statistics and
aggregate totals.

## Output

If `--out` is omitted, each paper's reconstruction is written under its crop
root:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      native/
      parsed/
      predictions/
      batch_summary.json
```

`native/`
: Adapter-native evidence and provenance.

`parsed/`
: The common Tabulus structured representation derived from adapter-native
  output.

`predictions/`
: Raw reconstruction CSVs used for reconstruction evaluation and later
  Tabulus processing. These are pre-reference-resolution artifacts.

`batch_summary.json`
: The reconstruction batch manifest for one paper and one adapter.

For a single `--crops` input, `--out <directory>` is the exact reconstruction
output directory. For multiple crop roots, `--out <parent>` is a parent
directory and Tabulus writes each result beneath:

```text
<parent>/
  <crop-root-name>/
    <adapter>/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Different adapters have independent reconstruction directories and must not
overwrite each other's results.

For the full filesystem contract, see {doc}`../data-contracts/run-directory`.

## Prediction CSV Semantics

A prediction CSV is written only when:

1. the adapter result status is `ok`; and
2. exactly one structured table was parsed from that canonical crop.

If zero structured tables are available, Tabulus preserves the native and
parsed evidence but does not write a prediction CSV.

If multiple structured tables are parsed from one canonical crop, Tabulus
preserves the native and parsed evidence, does not choose one arbitrarily, does
not merge them automatically, and does not write a prediction CSV.

This is an intentional ambiguity-preservation rule. Prediction yield is not a
reconstruction-accuracy measure.

## Runtime Guidance

Runtime varies substantially with crop count, crop dimensions, model
initialization, cache state, hardware, and adapter/backend configuration.
Adapter-specific runtime observations are documented on the relevant External
Tools and current-state pages.

Operational runtime does not imply reconstruction accuracy, and prediction CSV
yield does not imply which adapter is scientifically preferable. Model quality
must be evaluated against gold-standard reconstruction data.

## Boundary

This stage does not perform reference-table classification, bibliography
extraction, reference matching, DOI resolution, final resolved CSV generation,
or continued-table merging.

After reconstruction, the next implemented stage is
{doc}`10-reference-table-classification`.
