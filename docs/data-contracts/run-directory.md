# Run Directory

This page is the authoritative filesystem contract for the current Tabulus
profiling and table-reconstruction stages. Directories appear as their
corresponding stages are run; a fresh paper directory will not contain every
layer immediately.

## Current Output Hierarchy

The current implemented commands write stage outputs next to the source PDFs
by default:

```text
<papers-directory>/
  tabulus-output/
    mineru/
      <backend>/
        <paper>/
          <MinerU-native run directory>/...
    table-crops/
      <paper>/
        tables_index.json
        images/
        reconstructions/
          <adapter>/
            native/
            parsed/
            predictions/
            batch_summary.json
```

`mineru/`
: Native MinerU document-processing output. Tabulus chooses the
  profiler/backend root, but MinerU owns the document/run hierarchy beneath
  it.

`table-crops/<paper>/`
: The canonical Tabulus handoff for physical tables detected in one paper.
  This is the stable interface between PDF profiling and table
  reconstruction.

`reconstructions/<adapter>/`
: One adapter-specific reconstruction result for the canonical crops in that
  paper. Different adapters use separate directories and must not overwrite or
  share result files.

`native/`
: Adapter-native reconstruction evidence and provenance. For PaddleOCR-VL,
  Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling
  TableFormer, Granite Vision 4.1 4B, TRivia-3B, GLM-OCR, Dolphin-v2,
  DeepSeek-OCR-2, Nanonets-OCR-s, MonkeyOCRv2-B-Parsing, and NVIDIA Nemotron
  Parse v1.2, this preserves the native representation returned or derived
  from the selected adapter.

`parsed/`
: Tabulus's common structured table representation derived from adapter-native
  output.

`predictions/`
: Raw reconstructed table CSVs. These are reconstruction predictions for
  evaluation and downstream processing, not DOI-resolved or
  bibliography-enriched final output.

`batch_summary.json`
: The reconstruction batch manifest for one paper and one adapter.

## Stage Dependencies

The current rebuilt pipeline is staged around persisted filesystem handoffs:

```text
PDF
  |
  v
native MinerU output
  |
  v
canonical table-crops/<paper>/
  |-- tables_index.json
  `-- images/
        |
        v
reconstructions/<adapter>/
  |-- native/
  |-- parsed/
  |-- predictions/
  `-- batch_summary.json
        |
        v
later reference-processing stages
```

Later stages may consume selected or reference-containing tables, but
reconstruction artifacts remain preserved for each physical table processed by
the reconstruction stage.

This separation also decouples ML environments. MinerU can run in one Python
or Conda environment, PaddleOCR-VL can run in another, Chandra OCR 2 can run
in another, NuExtract3 can run in another, Tesseract + Table Transformer can
run in another, RapidOCR + Docling TableFormer can run in another, Granite
Vision can run in another, TRivia-3B can run in another, GLM-OCR can run in
another, Dolphin-v2 can run in another, DeepSeek-OCR-2 can run in another,
Nanonets-OCR-s can run in another, and MonkeyOCRv2-B-Parsing can run in
another, and NVIDIA Nemotron Parse v1.2 can run in another.
The stable
contracts between stages are persisted files
such as `tables_index.json`, canonical crop images, reconstruction manifests,
and reconstruction artifacts.

## Current Profiling Output Convention

The current implemented `tabulus profile` command processes one PDF. When
`--out` is omitted, Tabulus chooses this profiling output root:

```text
<PDF directory>/
  tabulus-output/
    <profiler>/
      <resolved-backend>/
```

For the current MinerU workflow:

```text
<PDF directory>/
  tabulus-output/
    mineru/
      <resolved-backend>/
```

`mineru` is the profiler. `pipeline` and `hybrid-engine` are MinerU backends.
If `hybrid-engine` is requested but Tabulus falls back to `pipeline`, the
automatic output root uses the resolved backend name:

```text
tabulus-output/mineru/pipeline/
```

`--out` is an explicit profiling-root override. `--table-crops-out` separately
overrides the normalized table-crop handoff directory, and
`--no-export-table-crops` disables automatic crop export.

MinerU retains its native hierarchy under the profiler/backend root:

```text
tabulus-output/
  mineru/
    <resolved-backend>/
      <paper>/
        <MinerU-native run directory>/...
```

The exact files and subdirectories below `<MinerU-native run directory>` are
owned by MinerU and should not be treated as a stable Tabulus public schema.
Tabulus discovers the relevant MinerU result and derives the canonical
table-crop handoff from it. The stable interface for subsequent
table-processing stages is not the complete MinerU native directory; it is:

```text
tabulus-output/
  table-crops/
    <paper>/
      tables_index.json
      images/
```

Do not flatten or rename MinerU-native output files. The current Tabulus reader
recursively finds `*_content_list.json` and resolves table images from MinerU's
`img_path` values.

## Canonical Table-Crop Directory

The canonical crop handoff for one paper is:

```text
table-crops/
  <paper>/
    tables_index.json
    images/
      page_<page>_table_<table-id>.<ext>
      ...
```

`images/` contains one canonical crop per physical MinerU-detected table. The
source image extension is preserved where applicable. Table IDs identify
physical detected tables within the document; they are not necessarily the
printed table numbers in the paper.

Continued tables remain separate physical table crops. Tabulus does not merge
continued tables at this stage.

`tables_index.json` records the crop inventory and provenance needed by
downstream stages. Records include the physical `table_id`, page number,
canonical image path/name, bounding box when available, caption, footnote,
MinerU source image/path provenance, MinerU `table_body`, reference-section
positional information, and source identifier.

`mineru_table_body` is MinerU's own table reconstruction associated with that
detected table. The canonical crop image is the shared visual input that
external reconstruction adapters operate on. Do not treat MinerU `table_body`
as output from PaddleOCR-VL or another table-reconstruction adapter.

## Reconstruction Directory

The implemented `tabulus reconstruct-tables` command consumes one canonical
table-crop directory:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter paddleocr-vl \
  --device gpu:0
```

If `--out` is omitted, Tabulus writes one reconstruction tree for the selected
adapter under that crop root:

```text
<crop-root>/
  reconstructions/
    <adapter>/
```

If `--out <directory>` is provided with a single `--crops` input,
`<directory>` is the exact reconstruction output directory.

For multiple crop roots selected with `--crops-folder` or `--crops-list`,
`--out <parent>` is treated as a parent directory. Tabulus writes each paper
and adapter result under:

```text
<parent>/
  <crop-root-name>/
    <adapter>/
```

If `--out` is omitted, each paper uses:

```text
<paper-crop-root>/
  reconstructions/
    <adapter>/
```

The currently registered crop-consuming adapter directories are
`paddleocr-vl`, `chandra`, `nuextract3`, `tesseract-tatr`,
`rapidocr-tableformer`, `granite-vision-table`, `trivia`, `glm-ocr`,
`dolphin-v2`, `deepseek-ocr-2`, `nanonets-ocr-s`,
`monkeyocrv2-b-parsing`, and `nemotron-parse-v1-2`. Future adapters should
follow the same structure when implemented:

```text
reconstructions/
  <adapter>/
    native/
    parsed/
    predictions/
    batch_summary.json
```

### native/

`native/` preserves adapter-native evidence and provenance. Its purpose is
reproducibility and inspection:

- preserve what the reconstruction adapter produced
- avoid losing adapter-specific information during normalization
- allow later debugging or re-parsing

This is not the canonical Tabulus table schema.

### parsed/

`parsed/` is the Tabulus common structured representation derived from
adapter-native output. Normalization allows different reconstruction adapters
to expose table content through a common structure.

The parsed representation records table identity, adapter/model/device
metadata, source crop, status, parsed table count, one or more parsed table
structures, rows/cells, row and column dimensions, parser source, warnings, and
the prediction CSV path when one is written.

One physical crop can produce zero parsed tables, exactly one parsed table, or
multiple parsed tables. Tabulus preserves that ambiguity rather than silently
selecting an arbitrary table.

### predictions/

`predictions/` contains raw reconstructed CSV predictions. These CSVs are:

- reconstruction outputs
- suitable as inputs to reconstruction evaluation
- suitable as inputs to later Tabulus processing
- independent of reference-table classification

They are not DOI-resolved tables, bibliography-enriched final results, or
reference-matched final output. Do not delete prediction CSVs merely because a
table is later classified as non-reference-containing.

A prediction CSV is written only when reconstruction is successful and the
common parsed representation contains exactly one usable parsed table. If
reconstruction has status `error`, is empty, or is ambiguous because multiple
tables were parsed, Tabulus preserves the `native/` and `parsed/` evidence but
does not arbitrarily write a single prediction CSV.

### batch_summary.json

`batch_summary.json` is the reconstruction-stage manifest for one paper and
one adapter. It records the physical tables processed and links their
reconstruction artifacts and provenance, including table identity,
reconstruction status, native artifact, parsed artifact, prediction CSV when
written, adapter information, timings, and error text.

This manifest does not combine multiple papers into one scientific result.
Each paper retains its own reconstruction directory and batch manifest.

## Continued-Table Semantics

Continued tables remain separate physical entities throughout the currently
implemented stages:

- MinerU detection
- canonical crops
- reconstruction
- parsed artifacts
- prediction CSVs

A logical table spanning several pages can therefore correspond to several
physical table IDs and several reconstruction files. Automatic continued-table
merging is not implemented in the rebuilt pipeline.

## Reconstruction Reruns

A fresh reconstruction run for one adapter clears only current Tabulus-owned
reconstruction artifacts for that adapter before writing the new run:

```text
<crop-root>/reconstructions/<adapter>/
  native/
  parsed/
  predictions/
  batch_summary.json
```

The implemented cleanup removes and refreshes only:

- `native/`
- `parsed/`
- `predictions/`
- `batch_summary.json`
- `reference_table_classification.json`

It does not remove:

- `tables_index.json`
- `images/`
- MinerU-native profiling outputs
- reconstruction outputs belonging to sibling adapters
- unrelated files in the selected adapter directory
- other paper outputs

Input crop validation occurs before this cleanup for the implemented crop-root
manifest checks. For example, a missing or invalid `tables_index.json` prevents
the reconstruction rerun from clearing previous outputs.

## Reference-Table Classification Boundary

The current rebuilt `src/tabulus` library implements
`tabulus classify-reference-tables` downstream of reconstruction.

The default manifest is:

```text
<crop-root>/reconstructions/<adapter>/reference_table_classification.json
```

The classifier reads the common parsed representation and reconstruction
manifest. It does not overwrite `native/`, `parsed/`, `predictions/`, or
`batch_summary.json`. A non-reference classification means only that the table
does not proceed down the reference-resolution branch; it does not mean the
reconstruction is invalid.

The current rebuilt library does not yet implement bibliography extraction,
reference matching, DOI resolution, final resolved CSV generation,
continued-table merging, or a single complete `tabulus run` orchestrator.

For the future final DOI-enriched CSV contract, see {doc}`resolved-csv`.
