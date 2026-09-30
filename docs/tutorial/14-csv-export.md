# Step 7: Resolved CSV Export

## Goal

Join Step 6 paper-level scholarly identities back onto the physical table rows
linked in Step 5 and export user-facing resolved CSV files.

Step 7 is deterministic. It consumes existing Step 5 and Step 6 artifacts,
performs no Crossref, CORE, DOI, or LLM lookup, and does not change Step 2
prediction CSVs.

```text
Step 5 references/reference_matches.json
            \
             -> Step 7 resolved CSV export
            /
Step 6 references/reference_resolution.json
             |
             v
resolved_reference_tables/
```

## Inputs

Required inputs:

- one Step 5 `references/reference_matches.json` artifact
- one Step 6 `references/reference_resolution.json` paper-level registry

Optional input for continuation merging:

- Step 1 `tables_index.json`, either inferred from the canonical layout or
  supplied with `--tables-index`

Step 7 keeps the physical reconstructed table as the default unit of export.
Continuation-aware logical merging is opt-in and never removes the physical
resolved CSVs.

## CLI

```bash
tabulus export-resolved-csv \
  --reference-matches /path/to/reconstruction/references/reference_matches.json \
  --reference-resolution /path/to/artifact-root/references/reference_resolution.json \
  [--out /path/to/output-directory] \
  [--merge-continuations] \
  [--tables-index /path/to/tables_index.json]
```

`--reference-matches`
: Step 5 row-level reference matching artifact for one reconstruction output.

`--reference-resolution`
: Step 6 paper-level registry for the same source paper.

`--out`
: Optional resolved CSV output directory. If omitted, Tabulus writes
  `resolved_reference_tables/` inside the reconstruction directory inferred
  from `reference_matches.json`.

`--merge-continuations`
: Also create logical merged CSVs for explicit Step 1 continuation groups when
  deterministic compatibility checks succeed. Physical resolved CSVs are
  always retained.

`--tables-index`
: Optional explicit Step 1 `tables_index.json` path for continuation merging.
  This option requires `--merge-continuations`. In the canonical layout,
  Tabulus normally infers the file from the reconstruction directory.

## Output

By default, Step 7 writes:

```text
<reconstruction>/
  resolved_reference_tables/
    <prediction-stem>_resolved.csv
    resolved_tables.json
```

Each physical resolved CSV preserves the reconstructed rows and appends
resolution/provenance columns. Multi-reference cells are encoded as compact
JSON arrays so the bibliography-index, status, DOI, title, author, year, venue,
source, confidence, reason, raw-reference, and unmatched-token values remain
positionally aligned.

Prediction CSVs remain unchanged. Rejected Step 6 decisions and Step 5
unmatched tokens remain visible rather than being silently discarded.

See {doc}`../data-contracts/resolved-csv` for the full output contract.

## Continuation Merging

Continued physical tables are not merged by default. With
`--merge-continuations`, Step 7 uses explicit Step 1 continuation relationships
and rechecks reconstructed-table compatibility at export time.

A continuation relationship does not guarantee that merging will succeed.
Known continuation groups are merged only when the group is complete in the
physical Step 7 exports and the reconstructed tables pass deterministic column
compatibility checks. Incomplete or structurally incompatible groups remain as
separate physical resolved CSVs, with the reason recorded in
`resolved_tables.json`.

Successful logical merges are additional files beneath:

```text
resolved_reference_tables/merged/
```

Merged CSVs append:

```text
tabulus_physical_table_id
```

so every merged row remains traceable to its physical source table.

## Examples

### TabulusBench

These examples follow the same P4 convention as the preceding tutorial pages.
They assume earlier steps have already produced a P4 reconstruction,
`reference_matches.json`, and a Step 6 `reference_resolution.json`.

Set portable roots:

```bash
export TABULUS_WORK="/path/to/tabulus-work"

P4_RECON="$TABULUS_WORK/P4/reconstructions/tesseract-tatr"
P4_ARTIFACT_ROOT="$TABULUS_WORK/stage4/Biomedicine_And_Health/clinical_research/P4"
```

`P4_RECON` is the Step 2/3/5 reconstruction directory for one adapter.
`P4_ARTIFACT_ROOT` is the paper-level artifact root containing the Step 4
bibliography and the Step 6 resolution registry.

### 1. Export resolved CSVs for one paper

```bash
tabulus export-resolved-csv \
  --reference-matches "$P4_RECON/references/reference_matches.json" \
  --reference-resolution "$P4_ARTIFACT_ROOT/references/reference_resolution.json"
```

The default output is:

```text
$P4_RECON/
  resolved_reference_tables/
    <prediction-stem>_resolved.csv
    resolved_tables.json
```

Use `--out` when you want the resolved CSVs somewhere else:

```bash
tabulus export-resolved-csv \
  --reference-matches "$P4_RECON/references/reference_matches.json" \
  --reference-resolution "$P4_ARTIFACT_ROOT/references/reference_resolution.json" \
  --out "$TABULUS_WORK/P4/resolved/tesseract-tatr"
```

### 2. Export with continuation-aware logical merges

Use continuation merging only with a Step 1 `tables_index.json` that belongs to
the same physical table crops reconstructed by Step 2.

```bash
tabulus export-resolved-csv \
  --reference-matches /path/to/reconstruction/references/reference_matches.json \
  --reference-resolution /path/to/artifact-root/references/reference_resolution.json \
  --merge-continuations \
  --tables-index /path/to/table-crops/<paper>/tables_index.json
```

Physical resolved CSVs remain in `resolved_reference_tables/`. Compatible
logical outputs, if any, are additional files in
`resolved_reference_tables/merged/`.

### 3. Export many papers

There is no separate collection-level `export-resolved-csv` command. For a
corpus, invoke the same paper/adapter-level command for each existing Step 5
and Step 6 artifact pair. A portable manifest can contain:

```text
reference_matches_path,reference_resolution_path,out_dir
/path/to/P4/reconstructions/tesseract-tatr/references/reference_matches.json,/path/to/P4/references/reference_resolution.json,/path/to/P4/resolved/tesseract-tatr
```

Then run:

```bash
python - <<'PY'
from pathlib import Path
import csv
import subprocess

manifest = Path("/path/to/step7-manifest.csv")

with manifest.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for index, row in enumerate(reader, start=1):
        command = [
            "tabulus",
            "export-resolved-csv",
            "--reference-matches",
            row["reference_matches_path"],
            "--reference-resolution",
            row["reference_resolution_path"],
        ]
        out_dir = row.get("out_dir") or ""
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            command.extend(["--out", out_dir])

        print(f"[{index}] {row['reference_matches_path']}")
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            print(f"  failed with return code {result.returncode}")
PY
```

Keep generated Step 7 outputs outside immutable TabulusBench gold directories.

## Verification

Verify that:

1. physical resolved CSV row counts match their Step 2 predictions
2. Step 2 prediction CSVs are unchanged
3. every Step 5 linked bibliography index has a final Step 6 identity
4. rejected resolutions remain represented
5. multi-reference cells preserve positional JSON-array alignment
6. continuation merging never removes physical resolved files
7. merged rows retain their physical table provenance
