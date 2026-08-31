# Step 1: PDF Profiling

## Goal

Run MinerU on one or more scientific PDFs, discover physical table regions, and
write the canonical Tabulus table-crop handoff used by later reconstruction
stages.

MinerU is the current PDF profiler. It performs document/layout processing,
table localization, and native table extraction. Tabulus invokes MinerU,
discovers the relevant MinerU result files, exposes typed table-region metadata,
and exports canonical crop images plus `tables_index.json`.

## Input

`tabulus profile` supports three mutually exclusive PDF input modes:

- `--pdf <file>`: process one PDF
- `--folder <folder>`: process all PDFs directly inside a folder
- `--pdf-list <text-file>`: process PDFs listed in a UTF-8 text file

For `--folder`, discovery is non-recursive, only PDFs directly inside the
folder are processed, inputs are sorted deterministically by filename, and PDFs
are processed sequentially.

For `--pdf-list`, blank lines and lines beginning with `#` are ignored,
relative paths are resolved relative to the list file, and duplicate inputs are
rejected.

## CLI

Profile one PDF:

```bash
tabulus profile \
  --pdf "/path/to/paper.pdf" \
  --backend pipeline \
  --method auto
```

Profile all PDFs directly inside a folder:

```bash
tabulus profile \
  --folder "/path/to/papers" \
  --backend hybrid-engine \
  --method auto \
  --effort high
```

`mineru` is currently the only profiler. `pipeline` and `hybrid-engine` are
MinerU backends. If `hybrid-engine` is requested but Tabulus resolves to
`pipeline`, automatic output paths use the resolved backend name.

## Output

When `--out` is omitted, Tabulus chooses the profiling output root beside the
PDF:

```text
<PDF parent>/
  tabulus-output/
    mineru/
      <resolved-backend>/
```

MinerU owns the native document/run hierarchy below that root:

```text
tabulus-output/
  mineru/
    <resolved-backend>/
      <paper>/
        <MinerU-native run directory>/
          images/
          <paper>_content_list.json
          ...
```

Tabulus discovers the actual MinerU-native run directory after execution rather
than constructing it from `--method`.

By default, the same `tabulus profile` run also writes the canonical crop
handoff:

```text
tabulus-output/
  table-crops/
    <paper>/
      tables_index.json
      images/
```

Use `--table-crops-out PATH` to override that handoff directory, or
`--no-export-table-crops` to skip automatic crop export.

## Table-Region Metadata

The current library contract exposes typed table regions. A table region
preserves information such as:

```json
{
  "table_id": 1,
  "page_nr": 3,
  "image_path": "/path/to/table-crop.jpg",
  "mineru_img_path": "images/table_001.jpg",
  "bbox": [181, 60, 812, 130],
  "caption": [],
  "footnote": [],
  "table_body": "<table>...</table>",
  "in_references": false
}
```

`table_id` identifies a physical detected table within the document; it is not
necessarily the printed table number in the paper. `table_body` is MinerU's
native table reconstruction candidate, retained separately from later
crop-consuming reconstruction adapters.

## Standalone Crop Export

The standalone export command regenerates the canonical handoff from an
existing MinerU output directory without rerunning MinerU:

```bash
tabulus export-table-crops \
  --mineru-root "/path/to/tabulus-output/mineru/<backend>/<paper>/<run-dir>" \
  --out "/path/to/tabulus-output/table-crops/<paper>"
```

The export writes:

```text
<crop-root>/
  tables_index.json
  images/
    page_<page>_table_<table-id>.<ext>
```

The exporter preserves the original MinerU image extension instead of
converting every crop to PNG.

## Boundary

PDF profiling does not perform crop-consuming table reconstruction,
reference-table classification, bibliography extraction, reference matching,
DOI resolution, final resolved CSV generation, or continued-table merging.

## Common Failure Modes

| Failure | Likely cause | Fix |
| --- | --- | --- |
| File not found | Wrong PDF, list, or output path | Validate paths before processing. |
| MinerU output missing | MinerU failed or did not write `*_content_list.json` | Inspect `mineru_stderr.log` and `tabulus_run.txt` where available. |
| No table regions | MinerU found no table entries or image provenance cannot be resolved | Inspect MinerU `*_content_list.json` and `img_path` values. |
| Incorrect table crop | MinerU detected the wrong region or reading order | Inspect MinerU layout/debug output beside the native run. |
| Weak structured table | MinerU `table_body` is incomplete or malformed | Compare it against crop-consuming reconstruction adapters before choosing an output for evaluation. |

## Next Step

After PDF profiling and canonical crop export, run
{doc}`08-table-ocr` on the canonical MinerU table crops.
