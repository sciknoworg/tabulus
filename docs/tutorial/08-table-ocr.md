# Step 2: Table Reconstruction

Table reconstruction turns the canonical MinerU crops from Step 1 into
structured table artifacts.

This stage is model-independent. An adapter may use OCR, a document
vision-language model, table-structure recognition, or another reconstruction
method, but every crop-consuming adapter receives the same canonical MinerU
crop.

## Input

Run this stage on one canonical table-crop root or on a collection of crop
roots:

```text
<crop-root>/
  tables_index.json
  images/
```

`tables_index.json` records the physical MinerU-detected tables, crop image
paths, page numbers, bounding boxes, captions, footnotes, MinerU provenance,
and MinerU `table_body` values where available.

The comparison boundary is:

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

It is not a comparison of separate PDF table detectors. Adapters must not go
back to the original PDF to choose their own table regions or crops.

Each physical crop is reconstructed independently. Continued-table merging,
bibliography extraction, reference matching, and DOI resolution are outside
this stage.

## Supported Adapters

MinerU `table_body` is retained as a native reconstruction candidate produced
during PDF profiling. It is not a crop-consuming `tabulus.table_ocr` adapter.

Use one of these values for `--adapter`:

```text
paddleocr-vl             chandra                  tesseract-tatr
rapidocr-tableformer     nuextract3               granite-vision-table
trivia                   glm-ocr                  dolphin-v2
deepseek-ocr-2           nanonets-ocr-s           monkeyocrv2-b-parsing
nemotron-parse-v1-2      hunyuanocr-1-5           dots-mocr
internvl3-5-8b
```

`paddleocr-vl`, `chandra`, `tesseract-tatr`, and `rapidocr-tableformer`
support CPU and GPU devices in the registry. The other current adapters are
registered as GPU-only in the validated Tabulus configuration.

Adapter-specific model revisions, prompts, runtime versions, and limitations
belong in the External Tools pages. The software interface and batch contract
are described in {doc}`../modules/table-ocr-adapters`.

## CLI

### Create Crop Roots

For one PDF, Step 1 creates one crop root:

```bash
tabulus profile \
  --pdf "/path/to/paper.pdf" \
  --backend pipeline
```

For several PDFs in one folder:

```bash
tabulus profile \
  --folder "/path/to/papers" \
  --backend hybrid-engine \
  --method auto \
  --effort high
```

For an explicit UTF-8 list of PDFs:

```bash
tabulus profile \
  --pdf-list "/path/to/pdfs.txt" \
  --backend hybrid-engine \
  --method auto \
  --effort high
```

By default, profiling writes per-paper crop roots under:

```text
<PDF directory>/tabulus-output/table-crops/<paper>/
```

### Reconstruct One Paper

Use `--crops` for one canonical crop root:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter <adapter> \
  --device gpu:0
```

### Reconstruct Several Papers

Use `--crops-folder` when each immediate child directory is one paper crop
root:

```bash
tabulus reconstruct-tables \
  --crops-folder "/path/to/tabulus-output/table-crops" \
  --adapter <adapter> \
  --device gpu:0
```

Use `--crops-list` when crop roots are listed explicitly:

```bash
tabulus reconstruct-tables \
  --crops-list "/path/to/crop-roots.txt" \
  --adapter <adapter> \
  --device gpu:0
```

The crop input modes are mutually exclusive:

- `--crops <crop-root>`: process one canonical table-crop directory.
- `--crops-folder <folder>`: process immediate child directories containing
  `tables_index.json`.
- `--crops-list <text-file>`: process crop roots listed in a UTF-8 text file.

For `--crops-folder`, discovery is non-recursive and sorted by directory name.
For `--crops-list`, blank lines and lines starting with `#` are ignored,
relative paths are resolved relative to the list file, and duplicate crop roots
are rejected.

## Output

If `--out` is omitted, each paper's reconstruction is written below its crop
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

For the full filesystem contract, see {doc}`../data-contracts/run-directory`.

## Prediction CSV Rule

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

## Batch Behavior

For multiple crop roots, Tabulus:

- creates the selected adapter once
- reuses that adapter instance across the complete command
- processes papers sequentially
- keeps each paper's reconstruction outputs isolated
- reports per-paper statistics and aggregate totals

Different adapters write to independent reconstruction directories and must
not overwrite each other's artifacts.

## Runtime Guidance

Runtime varies with crop count, crop dimensions, model initialization, cache
state, hardware, and adapter/backend configuration. Treat runtime observations
as engineering guidance, not model-quality evidence.

Adapter-specific runtime and environment notes live on the External Tools and
GPU installation pages. Scientific reconstruction quality must be measured
against gold-standard table annotations.

## Next Stage

After reconstruction, run the implemented reference-table classifier:

```bash
tabulus classify-reference-tables \
  --reconstruction "/path/to/tabulus-output/table-crops/<paper>/reconstructions/<adapter>"
```

This stage writes `reference_table_classification.json` beside the
reconstruction artifacts. It does not overwrite `native/`, `parsed/`,
`predictions/`, or `batch_summary.json`.

Continue with {doc}`10-reference-table-classification`.
