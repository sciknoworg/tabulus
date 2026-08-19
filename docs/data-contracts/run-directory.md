# Run Directory

The long-term normalized Tabulus pipeline should give each paper one run directory. This is the future shared filesystem contract between modules, not a description of every directory produced by the current partial implementation:

```text
runs/
  <paper>/
    input/
      paper.pdf
    metadata/
      pdf_profile.json
      reference_section.json
    profiling/
      mineru/
        <resolved-backend>/
          <MinerU-native document/run hierarchy>
    tables/
      crops/
      reconstructions/
        <adapter>/
          native/
          parsed/
          predictions/
          batch_summary.json
    references/
      bibliography.json
      reference_matches.json
    resolved_reference_tables/
    evaluation/
    report/
```

Adapter-owned source outputs should be retained inside the run directory instead of being treated as the normalized contract. For the current MinerU adapter, keep the full MinerU output directory, including `content_list.json`, `layout.pdf`, `middle.json`, `model.json`, reconstructed Markdown, and generated images.

Downstream modules should consume normalized Tabulus outputs, such as `metadata/pdf_profile.json`, indexed table images, normalized reconstructions, and prediction CSV files. Debugging and evaluation should be able to trace those outputs back to the original adapter-native files without mutating them.

The important artifact layers are:

- native/intermediate pipeline artifacts from external tools, such as MinerU output files or PaddleOCR native result views
- parsed and normalized reconstruction artifacts used by Tabulus components
- prediction CSV files used for table-reconstruction evaluation
- bibliography and reference-matching artifacts used for DOI enrichment
- resolved CSV files produced after reference matching and DOI resolution

## Current Profiling Output Convention

The current implemented `tabulus profile` command uses a simpler profiling-output convention when `--out` is omitted:

```text
<PDF directory>/
  tabulus-output/
    <profiler>/
      <resolved-backend>/
    table-crops/
      <PDF stem>/
```

For the current MinerU CPU path:

```text
<PDF directory>/
  tabulus-output/
    mineru/
      pipeline/
    table-crops/
      <PDF stem>/
```

`mineru` is the profiler. `pipeline` and `hybrid-engine` are MinerU backends.

If `hybrid-engine` is requested but Tabulus falls back to `pipeline`, the automatic output directory uses the resolved backend name:

```text
tabulus-output/mineru/pipeline/
```

`--out` remains available for explicit profiling-root override. `--table-crops-out` separately overrides the normalized table-crop handoff directory, and `--no-export-table-crops` disables automatic crop export.

MinerU retains its own native output hierarchy under the profiler/backend root, typically:

```text
tabulus-output/
  mineru/
    <resolved-backend>/
      <document>/
        <MinerU-native run directory>/
          images/
          <document>_content_list.json
          <document>_content_list_v2.json
          <document>_layout.pdf
          <document>_middle.json
          <document>_model.json
          <document>_origin.pdf
          <document>.md
```

Do not flatten or rename MinerU-native output files. The current Tabulus reader recursively finds `*_content_list.json` and resolves table images from MinerU's `img_path` values. The normalized table-crop handoff copies those canonical MinerU crops into:

```text
tabulus-output/
  table-crops/
    <PDF stem>/
      tables_index.json
      images/
```

## Current Table-Reconstruction Output Convention

The implemented `tabulus reconstruct-tables` command consumes the table-crop handoff root:

```text
tabulus-output/
  table-crops/
    <PDF stem>/
      tables_index.json
      images/
```

For:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter paddleocr-vl \
  --device gpu:0
```

if `--out` is omitted, Tabulus writes one reconstruction tree for the selected adapter under that crop root:

```text
<crop-root>/reconstructions/<adapter>/
```

The currently registered adapter directory is `paddleocr-vl`. The same namespace design allows other adapters to have separate directories later, if they are implemented behind the same reconstruction contract.

```text
tabulus-output/
  table-crops/
    <PDF stem>/
      tables_index.json
      images/
        page_<page>_table_<id>.<ext>
        ...
      reconstructions/
        <adapter>/
          native/
          parsed/
          predictions/
          batch_summary.json
```

`tables_index.json`
: The canonical manifest of physical tables produced from MinerU table discovery and crop export. It links each table crop to `table_id`, page number, crop image path, bounding box, caption, footnote, MinerU provenance, and MinerU `table_body` when available. This is the authoritative handoff between MinerU table localization and table reconstruction.

`images/`
: The canonical MinerU-generated table crops. Reconstruction adapters should consume these same images rather than re-detecting or re-cropping tables from the source PDF.

`reconstructions/<adapter>/`
: The reconstruction namespace for one adapter. Each adapter gets its own output directory so multiple reconstruction approaches can process the same canonical MinerU crops without overwriting each other.

`native/`
: Adapter-native output and provenance for each crop. This layer supports reproducibility, debugging, preservation of original model/tool results, and investigation of failed or unusual parses. Downstream Tabulus stages should not need to depend directly on adapter-specific native formats.

`parsed/`
: Tabulus's common structured table representation. This bridges adapter-specific output and downstream Tabulus processing. It contains the parsed rectangular table plus metadata such as table identity, adapter/model/device, status, parsed table count, rows, row/column dimensions, parse source such as HTML or Markdown, warnings, and the prediction CSV path when one was written.

`predictions/`
: Pre-reference-resolution reconstructed table CSV files. These are suitable for reconstruction-quality evaluation against ground-truth CSVs and are the reconstructed cell values used by downstream Tabulus stages. Prediction CSVs must not later be overwritten with DOI-enriched or reference-resolved values; final DOI-enriched tables belong in a separate downstream artifact location.

`batch_summary.json`
: The manifest for one reconstruction batch. Users and programs should start here after a batch run. It records batch-level fields such as adapter, input crop root, output directory, tables requested, tables OK, tables empty, tables error, prediction CSV count, total elapsed time, and summary path. It also records per-table fields such as `table_id`, source crop path, status, elapsed time, parsed table count, native result path, parsed result path, prediction CSV path, and error text.

Filename stems such as `page_006_table_001.csv` describe the physical crop:

- page 6
- Tabulus physical `table_id` 1

The Tabulus `table_id` is an internal physical-table identifier derived from the MinerU discovery sequence. It is not necessarily the table number printed in the scientific article. Each physical MinerU crop remains independent through reconstruction; continued-table merging is not currently performed at this stage.

## Reconstruction Reruns

The intended default rerun contract is that a new reconstruction run for a selected adapter starts from a clean set of Tabulus-owned reconstruction artifacts for that adapter:

```text
<crop-root>/reconstructions/<adapter>/
  native/
  parsed/
  predictions/
  batch_summary.json
```

That cleanup must not remove:

- `tables_index.json`
- `images/`
- MinerU-native profiling outputs
- reconstruction outputs belonging to other adapters

For safety, cleanup should apply only to Tabulus-owned reconstruction artifacts inside the selected adapter output. It should not blindly delete arbitrary contents of a user-supplied `--out` directory.

Current implementation note: cleanup of stale adapter artifacts is not implemented yet. The current batch writer creates the adapter output directory and overwrites same-named native, parsed, prediction, and summary files, but it does not remove stale files left from an earlier run if the table set or filenames change.

This reconstruction output is still upstream of reference classification, bibliography extraction, reference matching, DOI resolution, continued-table merging, and final resolved CSV export.
