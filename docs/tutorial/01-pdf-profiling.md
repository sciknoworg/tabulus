# Step 1: PDF Profiling

PDF profiling is the first runnable Tabulus step. It takes scientific PDF
files, runs a PDF profiler, discovers table regions, and exports those detected
tables as canonical crop images for Step 2 table reconstruction.

The current implemented profiler is MinerU. `tabulus profile` is the Tabulus
interface; `--backend`, `--method`, and `--effort` select MinerU execution
behavior behind that interface.

## What This Step Creates

Step 1 creates two conceptual output areas for each profiled paper:

```text
PDF
  |
  +-- MinerU-native profiling artifacts
  |
  `-- Tabulus canonical table-crop handoff
        |-- tables_index.json
        `-- images/
```

The MinerU-native area keeps the profiler's document output and diagnostics.
The canonical crop handoff is the stable Tabulus interface for the next step:
Step 2 table reconstruction consumes this handoff.

Step 1 performs PDF profiling, table detection, and canonical crop export. It
does not perform crop-consuming table reconstruction, reference-table
classification, bibliography extraction, reference matching, scholarly
reference resolution, or resolved CSV export.

## CLI

The profile command has one required PDF input source plus optional MinerU and
output controls:

```bash
tabulus profile <one input mode> [MinerU options] [output options]
```

If `--backend` is omitted, Tabulus prompts interactively:

```text
1. pipeline       CPU-compatible [default]
2. hybrid-engine  GPU-accelerated
```

### Input Modes

Choose exactly one input mode.

| Mode | Behavior |
| --- | --- |
| `--pdf <file>` | Profiles one explicitly named PDF. |
| `--folder <folder>` | Profiles PDF files directly inside one folder, sorted by filename. Discovery is non-recursive. |
| `--pdf-list <text-file>` | Profiles PDF paths listed one per line in a UTF-8 text file. Blank lines and lines beginning with `#` are ignored. Relative paths are resolved relative to the list file. Duplicate inputs are rejected. |

Use `--pdf-list` when the desired papers are nested across directories or when
you need a reproducible ordered input set.

### MinerU Options

| Option | Values | Meaning |
| --- | --- | --- |
| `--profiler` | `mineru` | PDF profiling tool. MinerU is currently the only implemented profiler. |
| `--backend` | `pipeline`, `hybrid-engine` | MinerU execution backend. |
| `--method` | `auto`, `txt`, `ocr` | MinerU parsing method. The default is `auto`. |
| `--effort` | `medium`, `high` | MinerU hybrid-engine effort. The default is `high`. |

`pipeline` is the CPU-compatible MinerU backend and is appropriate for Windows
or CPU-only environments. `hybrid-engine` is the GPU-accelerated MinerU backend.
If `hybrid-engine` is requested but the current Python environment does not
expose a suitable CUDA GPU, Tabulus reports the reason and falls back to
`pipeline`.

Tabulus passes `--effort` to MinerU only when the resolved backend is
`hybrid-engine`. The current profiling workflow always asks MinerU to extract
tables and disables formula extraction and image analysis internally; those are
not currently Tabulus CLI options.

### Output Options

When output flags are omitted, Tabulus writes profiling outputs beside each
source PDF:

```text
<PDF directory>/tabulus-output/
  mineru/<resolved-backend>/...
  table-crops/<PDF stem>/...
```

`--out <path>`
: Sets the MinerU profiling output root. For multiple PDFs, the same profiling
  root is used and MinerU writes each document under its own PDF-stem
  subdirectory.

`--table-crops-out <path>`
: Sets the canonical crop handoff location. For one PDF, this is the exact crop
  root. For multiple PDFs, this is a parent directory and Tabulus writes one
  crop root per PDF stem below it.

`--no-export-table-crops`
: Skips the automatic canonical crop handoff and keeps only the MinerU-native
  profiling output.

Custom output directory names are user-defined and do not indicate which
MinerU backend ultimately executed; Tabulus reports the resolved backend in
its diagnostics.

### Reusing Existing MinerU Output

If MinerU has already run, use `export-table-crops` to regenerate the canonical
crop handoff without profiling the PDF again:

```bash
tabulus export-table-crops \
  --mineru-root "/path/to/tabulus-output/mineru/<backend>/<paper>/<run-dir>" \
  --out "/path/to/tabulus-output/table-crops/<paper>"
```

The exporter preserves the original MinerU image extension instead of
converting every crop to PNG.

## Output Structure and Step Handoff

Tabulus owns the profiling root passed through `--out` or, by default, the
per-PDF `tabulus-output/mineru/<resolved-backend>/` directory. MinerU owns the
document hierarchy below that root and chooses the native run directory name.

A typical public Step 1 output has this shape:

```text
<PDF directory>/
  tabulus-output/
    mineru/
      <resolved-backend>/
        <paper>/
          <MinerU-native run directory>/
            ...
            mineru_stdout.log
            mineru_stderr.log
            tabulus_run.txt
    table-crops/
      <paper>/
        tables_index.json
        images/
          page_<page>_table_<table-id>.<ext>
```

With custom `--out` or `--table-crops-out` values, the top-level roots are the
paths supplied on the command line, but the same division remains: MinerU owns
the native profiling hierarchy, and Tabulus owns the canonical crop handoff.

After MinerU succeeds, Tabulus locates the actual native run directory from the
generated `*_content_list.json` file rather than assuming a fixed MinerU folder
name. Diagnostic logs are written beside the native output when possible; if
MinerU fails before a native run directory can be identified, diagnostics may
be written at the document level instead.

The canonical crop handoff is the Step 2 input. `tables_index.json` records
the detected crop inventory and provenance, including physical `table_id`, page
number, crop image name, bounding box when available, caption, footnote, MinerU
source image/path provenance, MinerU `table_body`, reference-section position
information, and source identifier where available.

For file-level details, see {doc}`../data-contracts/mineru-output-files` for
MinerU-native artifacts and {doc}`../data-contracts/tables-index-json` for the
Tabulus crop manifest.

`table_id` identifies a physical table detected by the profiler within the
document. It is not necessarily the table number printed in the paper. MinerU
`table_body` is MinerU's native table reconstruction candidate; the canonical
crop image is the shared visual input for crop-consuming reconstruction
adapters.

## Common Failure Modes

| Failure | Likely cause | Fix |
| --- | --- | --- |
| File not found | Wrong PDF, list, or output path | Validate the selected input mode and paths. |
| No PDFs found | `--folder` was pointed at a directory without direct PDF children | Use a folder that directly contains PDFs, or create a `--pdf-list`. |
| Duplicate PDF input | The same resolved PDF appears more than once in a list | Remove duplicate entries from the list. |
| MinerU output missing | MinerU failed or did not write `*_content_list.json` | Inspect `mineru_stderr.log` and `tabulus_run.txt` where available. |
| No table regions | MinerU found no table entries or image provenance cannot be resolved | Inspect MinerU `*_content_list.json` and source image paths. |
| Incorrect table crop | MinerU detected the wrong region or reading order | Inspect MinerU layout/debug output beside the native run. |

## Examples

### TabulusBench

[TabulusBench](https://zenodo.org/records/20230340) is the benchmark dataset
used for concrete tutorial examples. Throughout the tutorial, `P4` is used when
a concrete TabulusBench one-paper example is needed:

- paper ID: `P4`
- domain: `Biomedicine_And_Health`
- subdomain: `clinical_research`
- PDF: `Biomedicine_And_Health/clinical_research/P4/P4.pdf`

Set portable roots before running the examples:

```bash
export TABULUSBENCH_ROOT="/path/to/tabulusbench"
export TABULUS_WORK="/path/to/tabulus-work"
P4_PDF="$TABULUSBENCH_ROOT/Biomedicine_And_Health/clinical_research/P4/P4.pdf"
```

A one-paper run means processing the complete relevant input for one paper. For
Step 1, that means the original `P4.pdf`. A full TabulusBench run means every
original paper PDF in the benchmark. Later steps may use different
step-specific inputs.

Benchmark note: `P4/reference_tables/` is immutable TabulusBench gold, with six
annotated reference-containing tables and adjacent `gold.csv` files:

```text
P4/
  reference_tables/
    tables_index.json
    tables/
      page_004_table_001/
      ...
      page_009_table_006/
```

Step 1 starts from `P4.pdf` and writes newly generated detected table crops to
the selected Tabulus output area. It must not write into, replace, or
regenerate `P4/reference_tables/`, and the Step 1 detected table set may
differ from the six gold reference-containing tables selected for benchmark
annotation.

#### One paper: P4

The input is the original P4 PDF:

```text
P4.pdf
  |
  v
MinerU profiling output
  +
Tabulus canonical detected table crops
```

CPU-compatible run:

```bash
tabulus profile \
  --pdf "$P4_PDF" \
  --backend pipeline \
  --method auto \
  --out "$TABULUS_WORK/P4/profiling/cpu" \
  --table-crops-out "$TABULUS_WORK/P4/table-crops-cpu"
```

GPU-accelerated run:

```bash
tabulus profile \
  --pdf "$P4_PDF" \
  --backend hybrid-engine \
  --method auto \
  --effort high \
  --out "$TABULUS_WORK/P4/profiling/gpu" \
  --table-crops-out "$TABULUS_WORK/P4/table-crops-gpu"
```

The CPU-compatible example writes outputs conceptually like this:

```text
$TABULUS_WORK/P4/
  profiling/cpu/
    P4/
      <MinerU-native run directory>/
        P4_content_list.json
        images/
        mineru_stdout.log
        mineru_stderr.log
        tabulus_run.txt
  table-crops-cpu/
    tables_index.json
    images/
      page_<page>_table_<table-id>.<ext>
```

The GPU-accelerated example uses the same input and step, but requests the
`hybrid-engine` backend. If the GPU backend is not available, Tabulus falls
back to `pipeline` and records the resolved backend in diagnostics.

#### Full TabulusBench

The benchmark hierarchy is nested by domain, subdomain, and paper. Do not use a
non-recursive `--folder` call on the benchmark root for a full benchmark run.
Use an explicit PDF list instead.

Create the list from the benchmark metadata:

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

The generated file is a normal `tabulus profile --pdf-list` input. It is not an
experiment manifest and does not modify benchmark gold.

CPU-compatible full-benchmark run:

```bash
tabulus profile \
  --pdf-list "$TABULUS_WORK/manifests/tabulusbench-pdfs.txt" \
  --backend pipeline \
  --method auto \
  --out "$TABULUS_WORK/full-tabulusbench/profiling/cpu" \
  --table-crops-out "$TABULUS_WORK/full-tabulusbench/table-crops-cpu"
```

GPU-accelerated full-benchmark run:

```bash
tabulus profile \
  --pdf-list "$TABULUS_WORK/manifests/tabulusbench-pdfs.txt" \
  --backend hybrid-engine \
  --method auto \
  --effort high \
  --out "$TABULUS_WORK/full-tabulusbench/profiling/gpu" \
  --table-crops-out "$TABULUS_WORK/full-tabulusbench/table-crops-gpu"
```

For multiple PDFs, `--out` is the shared MinerU profiling root and
`--table-crops-out` is a parent crop directory. Each processed paper keeps its
own PDF-stem subdirectory below those roots.
