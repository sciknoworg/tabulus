# Step 5: Reference Matching

## Goal

Step 5 links citation-bearing cells in Step 3-selected reconstructed tables to
entries in the ordered Step 4 bibliography. Step 3 answers which reconstructed
tables are reference-containing, Step 4 supplies the paper bibliography, and
Step 5 determines which bibliography positions are cited by each relevant table
cell.

The two branches meet here:

```text
Step 3: selected_reference_tables.json
          \
           \
            -> Step 5 reference matching
           /
          /
Step 4: references/bibliography.json

            |
            v

references/reference_matches.json
```

Step 5 is deterministic and offline. It links table citation occurrences to
bibliography positions; DOI validation and scholarly-identity resolution belong
to Step 6.

## Inputs

Step 5 consumes two production inputs:

1. `selected_reference_tables.json`
   : The Step 3 selection manifest. It identifies tables classified as
     reference-containing and points to their parsed Step 2 reconstruction
     artifacts.

2. `references/bibliography.json`
   : The Step 4 bibliography artifact extracted from the original PDF.

Step 5 does not read the original PDF, rerun table reconstruction, rerun
GROBID, or modify prediction CSVs.

## Single-Paper Usage

Run reference matching for one paper and one reconstruction adapter with:

```bash
tabulus match-references \
  --selected /path/to/paper/reconstructions/<adapter>/selected_reference_tables.json \
  --bibliography /path/to/paper/references/bibliography.json
```

If `--out` is omitted, Tabulus writes the production artifact inside the
reconstruction directory:

```text
/path/to/paper/reconstructions/<adapter>/references/reference_matches.json
```

To choose the output file explicitly:

```bash
tabulus match-references \
  --selected /path/to/paper/reconstructions/<adapter>/selected_reference_tables.json \
  --bibliography /path/to/paper/references/bibliography.json \
  --out /path/to/paper/reconstructions/<adapter>/references/reference_matches.json
```

The selected-table manifest and bibliography artifact must belong to the same
paper. Step 5 does not verify scholarly identity; it only links table-cell
reference expressions to bibliography positions.

## Worked Example

Given a reference cell:

```text
[12, 15-17]
```

Step 5 expands the numeric list and range to the one-based bibliography
positions:

```text
12, 15, 16, 17
```

A corresponding `reference_matches.json` row-level excerpt is:

```json
{
  "row_index": 4,
  "value": "[12, 15-17]",
  "found": true,
  "matched_reference_indices": [12, 15, 16, 17],
  "matched_references": [
    "Raw bibliography entry 12.",
    "Raw bibliography entry 15.",
    "Raw bibliography entry 16.",
    "Raw bibliography entry 17."
  ],
  "doi": ["", "10.1234/example", "", ""],
  "match_provenance": [
    {"reference_index": 12, "method": "numeric_position", "token": "12"},
    {"reference_index": 15, "method": "numeric_position", "token": "15"},
    {"reference_index": 16, "method": "numeric_position", "token": "16"},
    {"reference_index": 17, "method": "numeric_position", "token": "17"}
  ],
  "tokens_total": 4,
  "tokens_matched": 4,
  "unmatched_tokens": [],
  "is_header": false
}
```

`matched_references` and `doi` are copied from the Step 4 bibliography artifact.
Step 5 does not look up missing DOI values.

## Matching Behavior

The matcher records these method labels:

- `numeric_position`
- `doi_exact`
- `author_year`
- `author_only`
- `text_containment`

`numeric_position` interprets numeric table references as one-based positions
in the normalized Step 4 bibliography order stored in
`references/bibliography.json`. For example, `[12]` links to bibliography entry
12. This is positional linkage, not DOI enrichment.

Numeric normalization handles common lists and ranges, such as `[12, 14]`,
`[12-15]`, `88 and 89`, and `83, 90, and 91`. It also recovers conservative
OCR-spacing cases inside numeric-only cells. Textual author-year forms such as
`Smith (2020)` are not treated as numeric references.

DOI matching is exact against DOI values already present in Step 4 extraction
evidence. Author-based matching uses conservative author-year, author-only, and
text-containment fallbacks. Ambiguous textual matches may retain multiple
candidate bibliography entries rather than silently choosing one.

Step 5 first detects the reference column in the parsed table. Numeric markers
in non-reference content are not automatically bibliography citations. For
example, `mark[1]` in a measurement or footnote column is table-local content,
not a bibliography reference merely because it contains `[1]`.

## Running Across Many Papers / TabulusBench

There is no dedicated collection-level `tabulus match-references` command. For
corpus-scale production use, invoke the same paper-level command once for each
paper and adapter where both inputs exist:

- Step 3 `selected_reference_tables.json`
- Step 4 `references/bibliography.json`

A portable orchestration pattern is to prepare a UTF-8 manifest with one row per
paper/adapter pair:

```text
selected_path,bibliography_path,out_path
/path/to/P4/reconstructions/tesseract-tatr/selected_reference_tables.json,/path/to/P4/references/bibliography.json,/path/to/P4/reconstructions/tesseract-tatr/references/reference_matches.json
```

Then invoke the existing CLI for each row:

```bash
python - <<'PY'
from pathlib import Path
import csv
import subprocess

manifest = Path("/path/to/step5-manifest.csv")

with manifest.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for index, row in enumerate(reader, start=1):
        selected = row["selected_path"]
        bibliography = row["bibliography_path"]
        out = row.get("out_path") or ""

        command = [
            "tabulus",
            "match-references",
            "--selected",
            selected,
            "--bibliography",
            bibliography,
        ]
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            command.extend(["--out", out])

        print(f"[{index}] {selected}")
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            print(f"  failed with return code {result.returncode}")
PY
```

For TabulusBench, production Step 5 matching can be run wherever Step 3 and
Step 4 artifacts exist. Accuracy evaluation is narrower: the current controlled
Step 5 gold-standard evaluation covers only P251 and P252, not the full
252-paper benchmark. For other papers, coverage, unmatched-token, and
consistency statistics must not be called precision, recall, F1, or accuracy
unless suitable gold labels exist.

## Output And Downstream Boundary

`reference_matches.json` remains a table-cell-level artifact. It records the
reference column, row-level matches, matched bibliography indices, match
provenance, unmatched tokens, and skipped-table diagnostics. See
{doc}`../data-contracts/reference-matches-json` for the full production output
contract.

Step 6 consumes one or more Step 5 artifacts, takes the union of matched
bibliography indices across supplied reconstruction methods for a paper, and
deduplicates by:

```text
(paper, bibliography_index)
```

Step 5 does not resolve scholarly identity, query Crossref or CORE, call an
LLM, mutate Step 2 reconstruction files, or change the Step 4 bibliography
evidence.

## Skipped Tables

Step 5 skips a selected table instead of guessing when the referenced parsed
artifact contains:

- `no_parsed_table`
- `multiple_parsed_tables`

The skipped table is recorded in `skipped_tables`, and the rest of the matching
run can continue. Malformed input contracts or identity mismatches are treated
as errors.
