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
  this preserves the native representation returned or derived from the
  PaddleOCR-VL adapter.

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
or Conda environment, PaddleOCR-VL can run in another, and later adapters may
have their own environments. The stable contracts between stages are persisted
files such as `tables_index.json`, canonical crop images, reconstruction
manifests, and reconstruction artifacts.

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

If `--out <directory>` is provided with the current single-`--crops` command,
`<directory>` is the exact reconstruction output directory. The current CLI
does not yet provide multi-paper `--crops-folder` or `--crops-list` modes.

The currently registered adapter directory is `paddleocr-vl`. Future adapters
should follow the same structure when implemented:

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

## Reference-Processing Boundary

The current rebuilt `src/tabulus` library does not yet implement a
`classify-reference-tables` command, bibliography extraction, reference
matching, DOI resolution, final resolved CSV generation, continued-table
merging, or a single complete `tabulus run` orchestrator.

When a future reference-table classification stage is implemented, it should be
downstream of reconstruction and should not overwrite `native/`, `parsed/`,
`predictions/`, or `batch_summary.json`. A non-reference classification should
mean only that the table does not proceed down the reference-resolution branch;
it should not mean the reconstruction is invalid.

For the future final DOI-enriched CSV contract, see {doc}`resolved-csv`.
