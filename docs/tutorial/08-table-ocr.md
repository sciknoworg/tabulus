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
| DeepSeek OCR | Future | Not registered in the rebuilt library |

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

For multiple papers, process every immediate child directory that contains a
`tables_index.json` file:

```bash
tabulus reconstruct-tables \
  --crops-folder "/path/to/tabulus-output/table-crops" \
  --adapter tesseract-tatr \
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
