# Step 1: PDF Profiling

PDF profiling is the first runnable Tabulus stage. It takes one or more PDF
files, runs a PDF profiler, discovers physical table regions inside each PDF,
and exports those detected tables as canonical crop images for Stage 2
reconstruction.

The profiling interface is designed so different profiling tools can be used
behind the same Tabulus handoff. The current implemented profiler is MinerU.
That means `tabulus profile` is the Tabulus command, while `--backend`,
`--method`, and `--effort` select MinerU-specific behavior.

## Worked Example Convention

The canonical TabulusBench worked example is paper `P4`:

```text
Biomedicine_And_Health/clinical_research/P4/P4.pdf
```

Use portable roots in commands:

```bash
export TABULUSBENCH_ROOT="/path/to/tabulusbench"
export TABULUS_WORK="/path/to/tabulus-work"
P4_PDF="$TABULUSBENCH_ROOT/Biomedicine_And_Health/clinical_research/P4/P4.pdf"
```

`P4` has six annotated reference-containing tables under
`P4/reference_tables/`, and it has paper-level bibliography gold under
`P4/bibliography/gold.json`. Those are benchmark gold materials. Stage 1 starts
from the original `P4.pdf` and writes fresh profiling/crop outputs to a work
directory; it should not overwrite or regenerate the benchmark
`reference_tables/` directory.

Stage 1 table detection may find a different set of tables than the six
annotated reference-containing benchmark tables. The benchmark tables are gold
annotation inputs for later controlled examples and evaluation, not the
definition of all tables detected in the PDF.

CPU and GPU examples below are backend choices for the same stage and the same
input:

```text
same Tabulus stage
same input
        |
        +-- CPU-compatible MinerU pipeline backend
        |
        +-- GPU-accelerated MinerU hybrid-engine backend
```

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
  and copied canonical table crop images extracted from the table regions
  discovered by the profiler.

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

### P4 — CPU-Compatible Run

Profile the canonical `P4` PDF with the CPU-compatible MinerU `pipeline`
backend:

```bash
tabulus profile \
  --pdf "$P4_PDF" \
  --backend pipeline \
  --method auto \
  --out "$TABULUS_WORK/P4/profiling/mineru/pipeline" \
  --table-crops-out "$TABULUS_WORK/P4/table-crops"
```

This writes MinerU-native profiling output below the requested profiling root
and writes the Tabulus canonical crop handoff below
`$TABULUS_WORK/P4/table-crops`.

### P4 — GPU-Accelerated Run

Profile the same `P4` PDF with the GPU-accelerated MinerU `hybrid-engine`
backend:

```bash
tabulus profile \
  --pdf "$P4_PDF" \
  --backend hybrid-engine \
  --method auto \
  --effort high \
  --out "$TABULUS_WORK/P4/profiling/mineru/hybrid-engine" \
  --table-crops-out "$TABULUS_WORK/P4/table-crops-hybrid-engine"
```

If `hybrid-engine` is requested but the current environment does not expose a
suitable CUDA GPU, Tabulus reports the reason and falls back to `pipeline`.
`--effort` is passed to MinerU only when the resolved backend is
`hybrid-engine`.

### Full TabulusBench — PDF List

TabulusBench stores PDFs below domain and subdomain directories, so do not use
`--folder "$TABULUSBENCH_ROOT"` for a full benchmark run. `--folder` is
non-recursive and processes only PDFs directly inside one folder.

Create an explicit PDF list in the work directory from the benchmark metadata:

```bash
mkdir -p "$TABULUS_WORK/manifests"

python - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["TABULUSBENCH_ROOT"])
out = Path(os.environ["TABULUS_WORK"]) / "manifests" / "tabulusbench-pdfs.txt"

with (root / "metadata" / "papers.csv").open(newline="", encoding="utf-8") as handle:
    rows = csv.DictReader(handle)
    pdfs = [root / row["pdf_path"] for row in rows if row.get("pdf_path")]

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(str(path) for path in pdfs) + "\n", encoding="utf-8")
print(out)
PY
```

The generated list is a normal `tabulus profile --pdf-list` input: one PDF path
per line. It is not an experiment manifest and does not modify benchmark gold.

### Full TabulusBench — CPU-Compatible Run

Run Stage 1 over the complete PDF list with the CPU-compatible backend:

```bash
tabulus profile \
  --pdf-list "$TABULUS_WORK/manifests/tabulusbench-pdfs.txt" \
  --backend pipeline \
  --method auto \
  --out "$TABULUS_WORK/full-tabulusbench/profiling/mineru/pipeline" \
  --table-crops-out "$TABULUS_WORK/full-tabulusbench/table-crops"
```

### Full TabulusBench — GPU-Accelerated Run

Run the same complete PDF list with the GPU-accelerated backend:

```bash
tabulus profile \
  --pdf-list "$TABULUS_WORK/manifests/tabulusbench-pdfs.txt" \
  --backend hybrid-engine \
  --method auto \
  --effort high \
  --out "$TABULUS_WORK/full-tabulusbench/profiling/mineru/hybrid-engine" \
  --table-crops-out "$TABULUS_WORK/full-tabulusbench/table-crops-hybrid-engine"
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
reference-table classification, bibliography extraction, deterministic
reference matching, DOI resolution, final resolved CSV generation, or
continued-table merging.

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
