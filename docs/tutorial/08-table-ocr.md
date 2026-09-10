# Step 2: Table Reconstruction

Table reconstruction is the second runnable Tabulus step. It takes canonical
table crops and reconstructs each crop into structured table artifacts through
one table-reconstruction adapter.

Step 1 produces canonical table crops from PDF profiling and table detection.
Step 2 consumes those crops. It does not detect table regions in the original
PDF and does not perform reference-table classification, bibliography
extraction, reference matching, scholarly reference resolution, or resolved CSV
export.

## What This Step Creates

Step 2 creates one reconstruction output area for each crop root and selected
adapter:

```text
canonical table crops
  |
  +-- native adapter result
  +-- parsed Tabulus table representation
  `-- prediction CSV, when exactly one structured table is available
```

The native result preserves adapter evidence and provenance. The parsed result
is the common Tabulus representation. The prediction CSV is the raw
pre-reference-resolution table used by later steps and by table reconstruction
evaluation when gold CSV files are available.

An explicit empty reconstruction is a valid adapter outcome: it means the
adapter processed the crop but did not produce a usable structured table. It is
separate from an adapter or process error.

## CLI

The reconstruction command has one required crop input source, one selected
adapter, and an execution device:

```bash
tabulus reconstruct-tables <one input mode> --adapter <adapter> --device <device> [--out <directory>]
```

If `--adapter` is omitted, Tabulus uses `paddleocr-vl`. If `--device` is
omitted, Tabulus passes `cpu` to the adapter.

### Input Modes

Choose exactly one input mode.

| Mode | Behavior |
| --- | --- |
| `--crops <crop-root>` | Reconstructs every crop listed by one `tables_index.json`. |
| `--crops-folder <folder>` | Reconstructs every immediate child directory containing `tables_index.json`, sorted by directory name. Discovery is non-recursive. |
| `--crops-list <text-file>` | Reconstructs crop roots listed one per line in a UTF-8 text file. Blank lines and lines beginning with `#` are ignored. Relative paths are resolved relative to the list file. Duplicate crop roots are rejected. |

Use `--crops` for one paper, `--crops-folder` when crop roots are direct
children of one directory, and `--crops-list` when crop roots are nested across
domains, subdomains, or other project structure.

### Adapter and Device Options

Each `--adapter` value names a registered table-reconstruction method. The
current registry exposes:

| Adapter | Device support |
| --- | --- |
| `chandra` | CPU or GPU |
| `deepseek-ocr-2` | GPU |
| `dolphin-v2` | GPU |
| `dots-mocr` | GPU |
| `glm-ocr` | GPU |
| `granite-vision-table` | GPU |
| `hunyuanocr-1-5` | GPU |
| `internvl3-5-8b` | GPU |
| `monkeyocrv2-b-parsing` | GPU |
| `nanonets-ocr-s` | GPU |
| `nemotron-parse-v1-2` | GPU |
| `nuextract3` | GPU |
| `paddleocr-vl` | CPU or GPU |
| `rapidocr-tableformer` | CPU or GPU |
| `tesseract-tatr` | CPU or GPU |
| `trivia` | GPU |

Device support here is the implementation capability registered by Tabulus.
Specific benchmark protocols may choose particular hardware, but that hardware
choice is not part of the Step 2 CLI contract. A device string beginning with
`cpu` selects CPU execution where supported; a string beginning with `gpu`, such
as `gpu:0`, selects GPU execution where supported.

For adapter-specific model revisions, prompts, runtime dependencies, and usage
notes, see the External Tools pages and {doc}`../modules/table-ocr-adapters`.

### Output Options

If `--out` is omitted, reconstruction output is written below the crop root:

```text
<crop-root>/reconstructions/<adapter>/
```

For one `--crops` input, `--out <directory>` is the exact reconstruction output
directory. For multiple crop roots, `--out <parent>` is treated as a parent and
Tabulus writes each result below:

```text
<parent>/<crop-root-name>/<adapter>/
```

This built-in multi-root layout is useful when crop-root directory names are
unique. When many crop roots have the same leaf directory name, use repeated
single-root invocations with explicit per-paper `--out` paths so outputs do not
collide.

## Output Structure and Step Handoff

A typical Step 2 output has this shape:

```text
<reconstruction-output>/
  native/
    page_<page>_table_<table-id>.json
  parsed/
    page_<page>_table_<table-id>.json
  predictions/
    page_<page>_table_<table-id>.csv
  batch_summary.json
```

`native/`
: Stores the adapter-neutral `TableOCRResult`, including preserved
  adapter-native JSON or Markdown, adapter/model versions when available,
  device, source image, status, error text, and provenance.

`parsed/`
: Stores the common Tabulus parsed table payload. It records the result status,
  parsed table count, parsed rows, optional `prediction_csv` pointer, and any
  warnings.

`predictions/`
: Stores raw prediction CSV files. A prediction CSV is written only when the
  adapter status is `ok` and exactly one structured table was parsed from that
  canonical crop. If no table or multiple tables are parsed, Tabulus preserves
  the evidence in `native/` and `parsed/` without choosing an arbitrary CSV.

`batch_summary.json`
: Stores the one-paper, one-adapter batch summary: adapter name, display name,
  crop root, output directory, requested table count, `ok`, `empty`, and
  `error` counts, prediction CSV count, elapsed time, per-table artifact paths,
  and per-table errors when present.

The prediction CSV is the handoff for raw table reconstruction quality checks
and for later reference-processing steps. Table reconstruction can be evaluated
against gold CSV files using Relative Mapping Similarity (RMS); see
{doc}`../evaluation/table-extraction-quality` for the full evaluation contract.

For the filesystem data contracts, see {doc}`../data-contracts/tables-index-json`,
{doc}`../data-contracts/ocr-tables-json`, and
{doc}`../data-contracts/table-prediction-csv`.

## Common Failure Modes

| Failure | Likely cause | Fix |
| --- | --- | --- |
| `tables_index.json` missing | The input is not a canonical table-crop root | Point `--crops` at the directory that directly contains `tables_index.json`, or use the correct list/folder mode. |
| No crop roots found | `--crops-folder` was pointed at a directory without direct crop-root children | Use a folder whose immediate children contain `tables_index.json`, or provide a `--crops-list`. |
| Duplicate crop root | The same resolved crop root appears more than once in a list | Remove duplicate entries from the list. |
| Adapter does not support device | The selected adapter is not registered for the requested CPU/GPU mode | Choose a supported adapter/device pair from the registry table. |
| No prediction CSV for a crop | The adapter returned `empty`, returned `error`, or produced zero or multiple parsed tables | Inspect the matching `parsed/*.json`, `native/*.json`, and `batch_summary.json` item. |
| Output collision risk | Multiple crop roots share the same leaf name and are run with one shared multi-root `--out` parent | Use repeated `--crops` invocations with explicit per-paper output directories. |

## Examples

### TabulusBench

[TabulusBench](https://zenodo.org/records/20230340) is the benchmark dataset
used for concrete tutorial examples. Throughout the tutorial, `P4` is used when
a concrete TabulusBench one-paper example is needed:

- paper ID: `P4`
- domain: `Biomedicine_And_Health`
- subdomain: `clinical_research`
- Step 2 crop root: `Biomedicine_And_Health/clinical_research/P4/reference_tables`

Set portable roots before running the examples:

```bash
export TABULUSBENCH="/path/to/tabulusbench"
export TABULUS_WORK="/path/to/tabulus-work"
P4_CROPS="$TABULUSBENCH/Biomedicine_And_Health/clinical_research/P4/reference_tables"
```

For Step 2, a one-paper run means reconstructing all canonical crop inputs
belonging to one paper. For `P4`, the benchmark crop root contains six
annotated reference-containing table crops under `reference_tables/tables/`,
each with immutable benchmark `gold.csv` material. Step 2 reads the crop
images and `tables_index.json`; it must not overwrite or regenerate the
TabulusBench gold CSV files.

For a lightweight starting point, `tesseract-tatr` is a useful first adapter to
try. More model-heavy adapters may require GPU resources and additional runtime
dependencies, as described in their External Tools pages.

#### 1. Run one paper with one adapter

The input is the complete P4 benchmark crop root:

```text
P4/reference_tables/
  tables_index.json
  tables/
    page_004_table_001/crop.png
    ...
    page_009_table_006/crop.png
```

Run P4 with the `tesseract-tatr` table-reconstruction adapter on CPU:

```bash
tabulus reconstruct-tables \
  --crops "$P4_CROPS" \
  --adapter tesseract-tatr \
  --device cpu \
  --out "$TABULUS_WORK/P4/reconstructions/tesseract-tatr"
```

The output is one reconstruction directory for the selected paper and adapter:

```text
$TABULUS_WORK/P4/reconstructions/tesseract-tatr/
  native/
  parsed/
  predictions/
  batch_summary.json
```

The native JSON keeps adapter evidence and provenance. The parsed JSON keeps
the normalized Tabulus row/column representation and warnings. The prediction
CSV is written only for crops with one unambiguous structured table. The batch
summary records the per-crop statuses and artifact paths.

#### 2. Run one paper with multiple adapters

To compare reconstruction methods, keep the P4 input fixed and change only the
adapter and device/output selection. Keep each adapter's output in a separate
directory:

```bash
tabulus reconstruct-tables \
  --crops "$P4_CROPS" \
  --adapter tesseract-tatr \
  --device cpu \
  --out "$TABULUS_WORK/P4/reconstructions/tesseract-tatr"

tabulus reconstruct-tables \
  --crops "$P4_CROPS" \
  --adapter rapidocr-tableformer \
  --device cpu \
  --out "$TABULUS_WORK/P4/reconstructions/rapidocr-tableformer"

tabulus reconstruct-tables \
  --crops "$P4_CROPS" \
  --adapter paddleocr-vl \
  --device gpu:0 \
  --out "$TABULUS_WORK/P4/reconstructions/paddleocr-vl"

tabulus reconstruct-tables \
  --crops "$P4_CROPS" \
  --adapter granite-vision-table \
  --device gpu:0 \
  --out "$TABULUS_WORK/P4/reconstructions/granite-vision-table"
```

`tesseract-tatr`, `rapidocr-tableformer`, and `paddleocr-vl` are registered for
CPU or GPU execution. `granite-vision-table` is registered as GPU-only.

#### 3. Run one adapter on the full TabulusBench dataset

TabulusBench contains 250 papers and 540 benchmark-annotated
reference-containing table crops used as Step 2 reconstruction inputs. These
are reconstruction inputs, not historical Step 1 table-detection counts.

The dataset root includes `reconstruction_inputs.txt`, a list of the 250
paper-level `reference_tables` crop roots. Because those crop roots are nested
below domain, subdomain, and paper directories, and because they all share the
leaf name `reference_tables`, use repeated single-paper commands with explicit
per-paper output paths:

```bash
while IFS= read -r crop_root; do
  case "$crop_root" in ""|\#*) continue ;; esac
  paper_rel="${crop_root%/reference_tables}"

  tabulus reconstruct-tables \
    --crops "$TABULUSBENCH/$crop_root" \
    --adapter tesseract-tatr \
    --device cpu \
    --out "$TABULUS_WORK/stage2/tesseract-tatr/$paper_rel"
done < "$TABULUSBENCH/reconstruction_inputs.txt"
```

Each paper keeps an independent reconstruction output below its domain,
subdomain, and paper path, for example:

```text
$TABULUS_WORK/stage2/tesseract-tatr/Biomedicine_And_Health/clinical_research/P4/
  native/
  parsed/
  predictions/
  batch_summary.json
```

#### 4. Run all adapters on the full TabulusBench dataset

On a GPU-equipped system, every registered Step 2 adapter is registered for
GPU execution. This loop reconstructs the same 540 benchmark crop inputs with
every registered table-reconstruction method and keeps outputs separated by
adapter and paper:

```bash
ADAPTERS=(
  chandra
  deepseek-ocr-2
  dolphin-v2
  dots-mocr
  glm-ocr
  granite-vision-table
  hunyuanocr-1-5
  internvl3-5-8b
  monkeyocrv2-b-parsing
  nanonets-ocr-s
  nemotron-parse-v1-2
  nuextract3
  paddleocr-vl
  rapidocr-tableformer
  tesseract-tatr
  trivia
)

for adapter in "${ADAPTERS[@]}"; do
  while IFS= read -r crop_root; do
    case "$crop_root" in ""|\#*) continue ;; esac
    paper_rel="${crop_root%/reference_tables}"

    tabulus reconstruct-tables \
      --crops "$TABULUSBENCH/$crop_root" \
      --adapter "$adapter" \
      --device gpu:0 \
      --out "$TABULUS_WORK/stage2/all-adapters/$adapter/$paper_rel"
  done < "$TABULUSBENCH/reconstruction_inputs.txt"
done
```

For CPU-only environments, use the CPU-capable subset from the registry table:
`chandra`, `paddleocr-vl`, `rapidocr-tableformer`, and `tesseract-tatr`, and
pass `--device cpu`.

This full comparative setup is useful for benchmarking because every method
receives the same canonical table crops. The benchmark gold tables remain
read-only inputs for later evaluation; reconstruction outputs are written under
`$TABULUS_WORK`.
