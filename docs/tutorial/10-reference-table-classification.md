# Step 3: Reference-Table Classification

## Goal

Decide which reconstructed-table instances contain reference-like scientific
citation content and should enter the reference-processing branch.

This step is implemented in the rebuilt library as:

```bash
tabulus classify-reference-tables
```

## Input

Reference-table classification consumes reconstruction artifacts from one adapter:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      parsed/
      predictions/
      batch_summary.json
```

The classifier reads the common parsed table representation and the reconstruction batch manifest. It does not read the original PDF, rerun OCR, or modify prediction CSVs.

## Output

By default, the command writes:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      reference_table_classification.json
      selected_reference_tables.json
```

`reference_table_classification.json` records a routing/classification decision
for each reconstructed-table instance considered. `selected_reference_tables.json` is a
non-destructive pointer manifest containing only the tables selected for Step
5. Neither artifact overwrites:

- `native/`
- `parsed/`
- `predictions/`
- `batch_summary.json`

A non-reference classification means only that the table should not proceed
down the reference-processing branch. It does not mean the reconstruction is
invalid.

## CLI

Classify one reconstruction directory:

```bash
tabulus classify-reference-tables \
  --reconstruction "/path/to/table-crops/<paper>/reconstructions/<adapter>"
```

Classify all immediate crop roots beneath a table-crops parent for one adapter:

```bash
tabulus classify-reference-tables \
  --crops-folder "/path/to/tabulus-output/table-crops" \
  --adapter paddleocr-vl
```

Classify reconstruction directories listed in a UTF-8 text file:

```bash
tabulus classify-reference-tables \
  --reconstruction-list "/path/to/reconstructions.txt"
```

Blank lines and lines beginning with `#` are ignored. Relative paths in a
reconstruction list are resolved relative to the list file.

For multi-paper classification, the default manifest is written inside each selected reconstruction directory. `--out` is only valid when exactly one reconstruction directory is selected.

## Classification Model

One deterministic regex/rule classifier is applied independently to the outputs
of each table-reconstruction method. Every reconstructed-table instance is
classified independently first. The classifier uses the common parsed rows produced during reconstruction, preserves the legacy reference-bearing table heuristics, and records matched evidence.

The manifest includes fields such as:

- `is_reference_table`
- `independent_is_reference_table`
- `classification_source`
- `continued_from_table_id`
- `continuation_caption`
- `matched_header_cells`
- `matched_citation_cells`
- `reason`

Current heuristics include reference-like headers, citation-like cell content, DOI-like strings, author-year patterns, and conservative bare numeric references when those numbers occur inside explicitly reference-like columns such as `Refs.`, `References`, or `Citations`.

## Continued Tables

Continued-table handling is a separate layer on top of independent classification:

```text
reconstructed-table instance
  -> independent reference classification
  -> continuation relationship resolution
  -> final reference-table decision
```

An explicitly identified continuation may inherit a positive reference-table classification from its preceding logical table. The manifest preserves whether the final decision came from independent table evidence or continuation inheritance.

This does not merge files. Continued-table fragments retain separate crops and reconstruction artifacts
through parsing, prediction CSV export, and classification.

## Boundary

This step performs reference-table routing only. It does not extract bibliographies, match references, resolve DOI values, write resolved CSVs, merge continued tables, or run the complete end-to-end pipeline.

## Next Step

The next rebuilt branch is bibliography extraction, which produces
`references/bibliography.json` from the original PDF. It runs in parallel with
table processing and converges with classified reference-like tables at Step 5
reference matching. Step 6 resolves the union of referenced
bibliography indices once per paper; Step 7 exports resolved CSVs from Step 5
matches and the Step 6 paper-level registry.

## Examples

### TabulusBench

[TabulusBench](https://zenodo.org/records/20230340) is the benchmark dataset used for concrete tutorial examples. It currently contains 252 papers across five domains:

- `Biomedicine_And_Health`
- `Agriculture_Food_And_Environmental_Systems`
- `Computer_Science_AI_And_Data_Science`
- `Energy_Materials_Chemical_Sciences`
- `Engineering_Robotics_And_Built_Infrastructure`

The benchmark includes 605 expert-annotated positive physical reference-table fragments. `P4` is the same one-paper example used in the preceding tutorial steps:

- paper ID: `P4`
- domain: `Biomedicine_And_Health`
- subdomain: `clinical_research`
- benchmark crop root: `Biomedicine_And_Health/clinical_research/P4/reference_tables`

Step 3 operates on Step 2 reconstruction outputs rather than on the original PDF or canonical crop images directly. The benchmark gold under `reference_tables/` is read-only. Newly generated Step 3 artifacts belong in the reconstruction/output area, never in the benchmark gold directories.

Use portable roots for the examples:

```bash
export TABULUSBENCH="/path/to/tabulusbench"
export TABULUS_WORK="/path/to/tabulus-work"
```

### 1. Classify one paper

Start from an existing P4 reconstruction produced by Step 2, for example the Tesseract + TATR reconstruction used in the Step 2 tutorial:

```bash
P4_RECON="$TABULUS_WORK/P4/reconstructions/tesseract-tatr"
```

Classify all physical table reconstructions recorded by that Step 2 batch:

```bash
tabulus classify-reference-tables \
  --reconstruction "$P4_RECON"
```

This processes the complete P4 reconstruction directory for that adapter, not just one prediction CSV. The output directory keeps the original Step 2 artifacts and adds the two Step 3 manifests:

```text
$P4_RECON/
  native/
  parsed/
  predictions/
  batch_summary.json
  reference_table_classification.json
  selected_reference_tables.json
```

`reference_table_classification.json` contains one classification record per physical reconstructed table, including the evidence and final routing decision. `selected_reference_tables.json` is a non-destructive pointer manifest containing only the tables selected for the reference-processing branch.

Step 3 does not rewrite `native/`, `parsed/`, `predictions/`, or `batch_summary.json`. To inspect the full classification manifest:

```bash
python -m json.tool \
  "$P4_RECON/reference_table_classification.json"
```

### 2. Classify the full TabulusBench collection

For a full TabulusBench run, classify every paper-level Step 2 reconstruction directory. The benchmark hierarchy is nested by domain, subdomain, and paper, so use an explicit reconstruction list rather than assuming the reconstructions are immediate children of one flat folder.

The Step 2 full-benchmark examples write one adapter's output under:

```text
$TABULUS_WORK/stage2/tesseract-tatr/<domain>/<subdomain>/<paper>/
```

Build a UTF-8 reconstruction list from TabulusBench's `reconstruction_inputs.txt`. Each non-comment input line ends in `/reference_tables`; the corresponding reconstruction directory is the same paper-relative path under `$TABULUS_WORK/stage2/tesseract-tatr/`:

```bash
python - <<'PY'
from pathlib import Path
import os

bench = Path(os.environ["TABULUSBENCH"])
work = Path(os.environ["TABULUS_WORK"])
source = bench / "reconstruction_inputs.txt"
output = work / "manifests" / "tabulusbench-step3-tesseract-tatr.txt"
output.parent.mkdir(parents=True, exist_ok=True)

reconstruction_dirs = []
for raw in source.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    crop_rel = Path(line)
    if crop_rel.name != "reference_tables":
        raise ValueError(f"Expected a reference_tables input, got: {line}")
    paper_rel = crop_rel.parent
    reconstruction_dirs.append(work / "stage2" / "tesseract-tatr" / paper_rel)

output.write_text(
    "\n".join(str(path) for path in reconstruction_dirs) + "\n",
    encoding="utf-8",
)
print(f"Wrote {len(reconstruction_dirs)} reconstruction directories to {output}")
PY
```

The list should contain one reconstruction directory for each of the 252 TabulusBench papers, for example:

```text
/path/to/tabulus-work/stage2/tesseract-tatr/Biomedicine_And_Health/clinical_research/P4
...
/path/to/tabulus-work/stage2/tesseract-tatr/Energy_Materials_Chemical_Sciences/.../P252
```

Run the deterministic classifier over the list:

```bash
tabulus classify-reference-tables \
  --reconstruction-list \
  "$TABULUS_WORK/manifests/tabulusbench-step3-tesseract-tatr.txt"
```

This classifies the existing Step 2 outputs for all 252 paper-level reconstruction directories. Step 3 itself does not require GPU inference. Each paper receives its own `reference_table_classification.json` and `selected_reference_tables.json` inside its reconstruction directory. Do not pass `--out` for a multi-reconstruction invocation; the default per-reconstruction manifest locations are required.

### 3. Run Step 3 for another adapter

Step 3 is applied independently to every adapter's Step 2 output. The classifier is the same deterministic rule-based procedure regardless of which table-reconstruction adapter produced the parsed table.

For example, if PaddleOCR-VL reconstructions were written under:

```text
$TABULUS_WORK/stage2/paddleocr-vl/<domain>/<subdomain>/<paper>/
```

build the corresponding reconstruction list with `paddleocr-vl` in the output path and run the same command:

```bash
tabulus classify-reference-tables \
  --reconstruction-list \
  "$TABULUS_WORK/manifests/tabulusbench-step3-paddleocr-vl.txt"
```

TabulusBench's positive reference-table annotations can be used to evaluate positive-reference-table recovery for the benchmark fragments. When matching gold labels are unavailable for a comparison, use coverage, consistency, and agreement across reconstruction outputs rather than treating Step 3 routing as a generic accuracy result.
