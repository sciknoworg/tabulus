# Step 1: PDF Profiling

PDF profiling is the first runnable Tabulus stage. It takes one or more PDF
files, runs the current profiler, discovers physical table regions, and writes
the canonical table-crop handoff used by Stage 2 reconstruction.

The current profiler is MinerU. That means `tabulus profile` is the Tabulus
command, while `--backend`, `--method`, and `--effort` select MinerU behavior.

## What This Stage Creates

By default, one profiled paper produces two output areas beside the source PDF:

```text
<PDF parent>/
  tabulus-output/
    mineru/
      <resolved-backend>/
        <paper>/
          <MinerU-native run directory>/
            ...
    table-crops/
      <paper>/
        tables_index.json
        images/
```

`tabulus-output/mineru/<resolved-backend>/`
: The profiling output root that Tabulus gives to MinerU.

`<paper>/<MinerU-native run directory>/`
: MinerU's own native document/run hierarchy. MinerU chooses the final run
  directory name.

`tabulus-output/table-crops/<paper>/`
: The stable Tabulus handoff for Stage 2. This contains `tables_index.json`
  and copied canonical table crop images.

Later table reconstruction should use `table-crops/<paper>/`, not the full
MinerU-native directory.

## Input Modes

Choose exactly one PDF input mode:

| Mode | Use when |
| --- | --- |
| `--pdf <file>` | You want to profile one PDF. |
| `--folder <folder>` | You want every PDF directly inside one folder. |
| `--pdf-list <text-file>` | You want to control the PDF list explicitly. |

For `--folder`, discovery is non-recursive. Only PDFs directly inside the
folder are processed, inputs are sorted by filename, and papers run
sequentially.

For `--pdf-list`, use one PDF path per line. Blank lines and lines beginning
with `#` are ignored, relative paths are resolved relative to the list file,
and duplicate inputs are rejected.

## MinerU Options

`tabulus profile` currently exposes these MinerU-specific options:

| Option | Values | Meaning |
| --- | --- | --- |
| `--profiler` | `mineru` | PDF profiling tool. MinerU is currently the only profiler. |
| `--backend` | `pipeline`, `hybrid-engine` | MinerU execution backend. |
| `--method` | `auto`, `txt`, `ocr` | MinerU parsing mode. |
| `--effort` | `medium`, `high` | MinerU `hybrid-engine` processing effort. |

`--backend pipeline`
: CPU-compatible MinerU backend. Use this for Windows or CPU-only runs.

`--backend hybrid-engine`
: GPU-backed MinerU backend. If requested but GPU requirements are not met,
  Tabulus reports the reason and falls back to `pipeline`. Output paths use the
  resolved backend name.

`--method auto`
: Let MinerU choose text extraction or OCR handling.

`--method txt`
: Ask MinerU to use native PDF text extraction.

`--method ocr`
: Ask MinerU to use OCR.

`--effort high`
: Default effort for `hybrid-engine`. Tabulus passes `--effort` only when the
  resolved backend is `hybrid-engine`.

Tabulus also fixes these MinerU settings internally for the current profiling
workflow:

```text
table=True
formula=False
image_analysis=False
```

They are not currently Tabulus CLI arguments.

## CLI

Profile one PDF on a CPU-compatible backend:

```bash
tabulus profile \
  --pdf "/path/to/paper.pdf" \
  --backend pipeline \
  --method auto
```

Profile all PDFs directly inside a folder on the GPU backend:

```bash
tabulus profile \
  --folder "/path/to/papers" \
  --backend hybrid-engine \
  --method auto \
  --effort high
```

Profile PDFs from an explicit list:

```bash
tabulus profile \
  --pdf-list "/path/to/pdfs.txt" \
  --backend hybrid-engine \
  --method auto \
  --effort high
```

If you omit `--backend`, Tabulus prompts interactively:

```text
1. pipeline       CPU-compatible [default]
2. hybrid-engine  GPU-accelerated
```

## Output Controls

Most users can omit output flags and use the default per-paper layout.

Use `--out` only when you want to choose the MinerU profiling output root:

```bash
tabulus profile \
  --pdf "/path/to/paper.pdf" \
  --backend pipeline \
  --out "/path/to/profile-root"
```

MinerU still creates its native document/run hierarchy below that root.

Use `--table-crops-out` when you want to choose where the canonical crop
handoff is written:

```bash
tabulus profile \
  --folder "/path/to/papers" \
  --backend hybrid-engine \
  --table-crops-out "/path/to/table-crops"
```

For one PDF, `--table-crops-out` is the exact crop-root directory. For multiple
PDFs, it is treated as a parent directory and each paper receives its own
subdirectory:

```text
/path/to/table-crops/
  <paper-a>/
    tables_index.json
    images/
  <paper-b>/
    tables_index.json
    images/
```

Use `--no-export-table-crops` only when you want to keep the MinerU-native
profiling output but skip the Stage 2 handoff:

```bash
tabulus profile \
  --pdf "/path/to/paper.pdf" \
  --backend pipeline \
  --no-export-table-crops
```

## MinerU Native Run Directory

Tabulus owns only the profiling root:

```text
<PDF parent>/tabulus-output/mineru/<resolved-backend>/
```

MinerU owns the hierarchy below it:

```text
tabulus-output/
  mineru/
    <resolved-backend>/
      <paper>/
        <MinerU-native run directory>/
          images/
          <paper>_content_list.json
          <paper>_content_list_v2.json
          <paper>_layout.pdf
          <paper>_middle.json
          <paper>_model.json
          <paper>_origin.pdf
          <paper>.md
          mineru_stdout.log
          mineru_stderr.log
          tabulus_run.txt
```

After a successful MinerU run, Tabulus discovers the actual
`<MinerU-native run directory>` from the generated `*_content_list.json`
rather than predicting it from `--method`.

Validated MinerU 3.4.5 examples:

```text
pipeline/<paper>/auto/
hybrid-engine/<paper>/hybrid_auto/
```

These names are MinerU-owned behavior from tested configurations. They are not
Tabulus directory rules.

`mineru_stdout.log`, `mineru_stderr.log`, and `tabulus_run.txt` are Tabulus
diagnostic files written beside successful MinerU output. If MinerU fails
before a native run directory can be identified, diagnostics may be written at
the document level instead.

## Canonical Crop Handoff

The crop handoff is the stable interface for Stage 2:

```text
tabulus-output/
  table-crops/
    <paper>/
      tables_index.json
      images/
        page_<page>_table_<table-id>.<ext>
```

`tables_index.json` records the crop inventory and provenance needed by later
stages. It preserves physical `table_id`, page number, crop image name,
bounding box when available, caption, footnote, MinerU source image/path
provenance, MinerU `table_body`, reference-section position information, and
source identifier where available.

`table_id` identifies a physical detected table within the document. It is not
necessarily the printed table number in the paper.

MinerU `table_body` is MinerU's native table reconstruction candidate. The
canonical crop image is the shared visual input for Stage 2 reconstruction
adapters.

## Reuse Existing MinerU Output

If MinerU has already run, regenerate the canonical crop handoff without
profiling the PDF again:

```bash
tabulus export-table-crops \
  --mineru-root "/path/to/tabulus-output/mineru/<backend>/<paper>/<run-dir>" \
  --out "/path/to/tabulus-output/table-crops/<paper>"
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
| Weak structured table | MinerU `table_body` is incomplete or malformed | Compare it against Stage 2 reconstruction adapters before choosing an output for evaluation. |

## Next Step

After PDF profiling and canonical crop export, run
{doc}`08-table-ocr` on the canonical MinerU table crops.
