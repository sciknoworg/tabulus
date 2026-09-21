# Step 4: Bibliography Extraction

## Goal

Extract an ordered bibliography from the original scientific PDF using GROBID.
Step 4 is extraction only; it performs no Crossref, CORE, LLM, or
scholarly-identity resolution.

Bibliography extraction is a PDF-level branch. It runs in parallel with the
table-processing branch and does not consume MinerU table crops, reconstructed
prediction CSVs, or reference-table classification output.

```text
original PDF
      |
      +--> table detection / canonical crops
      |         |
      |         v
      |   table reconstruction
      |         |
      |         v
      |   structured-table parsing
      |         |
      |         v
      |   reference-table classification
      |
      +--> GROBID bibliography extraction
                |
                v
          references/bibliography.json

selected_reference_tables.json + bibliography.json
      |
      v
Step 5 reference matching
```

## Input

The implemented GROBID path consumes the original PDF directly.

`metadata/reference_section.json` may exist from PDF profiling and can remain
useful pipeline metadata, but it is not a required input to the current GROBID
bibliography extractor.

## Output

The normalized bibliography artifact is:

```text
<artifact-root>/
  references/
    bibliography.json
```

See {doc}`../data-contracts/bibliography-json` and {doc}`../external-tools/grobid`.

The current artifact records:

- one-based GROBID bibliography position
- preserved raw reference text
- DOI when one is already present in the extracted bibliography text
- extractor source
- title, authors, year, venue, volume, issue, and pages when GROBID supplies
  usable structured metadata

Missing scholarly metadata is not invented. Structured GROBID years take
precedence; if no structured year exists, Tabulus recovers a year only when
exactly one plausible year appears in the raw citation. Crossref enrichment,
CORE lookup, LLM adjudication, and scholarly identity resolution are outside
Step 4.

## Command Line

The normal Step 4 command is:

```bash
tabulus extract-bibliography \
  --pdf /path/to/paper.pdf \
  --out /path/to/artifact-root \
  --grobid-url http://localhost:8070
```

Arguments:

- `--pdf`: original scientific PDF
- `--out`: artifact root; Tabulus creates `references/bibliography.json`
  beneath it
- `--grobid-url`: GROBID service root
- `--timeout-seconds`: optional GROBID HTTP timeout

For a longer HTTP timeout:

```bash
tabulus extract-bibliography \
  --pdf /path/to/paper.pdf \
  --out /path/to/artifact-root \
  --grobid-url http://localhost:8070 \
  --timeout-seconds 300
```

The CLI processes one PDF per invocation. Collection-level orchestration is
outside Step 4.

## Python API

The CLI is a thin wrapper around the public Python API, which also processes
one PDF at a time:

```python
from pathlib import Path

from tabulus.bibliography.pipeline import extract_bibliography_artifact

extract_bibliography_artifact(
    Path("INPUT.pdf"),
    Path("OUTPUT_DIRECTORY"),
    grobid_url="http://localhost:8070",
)
```

This writes:

```text
OUTPUT_DIRECTORY/
  references/
    bibliography.json
```

## GROBID Service

Tabulus communicates with GROBID over HTTP. GROBID is an external service, not
a Python or Conda model dependency.

Pass the service root as `grobid_url`, commonly:

```text
http://localhost:8070
```

The client appends `/api/processReferences` and sends the original PDF as a
multipart request. It requests raw citations and disables GROBID citation
consolidation so external metadata lookup remains outside Step 4.

Check that GROBID is reachable before running extraction:

```bash
curl http://localhost:8070/api/isalive
```

In containerized environments, expose GROBID's HTTP port and keep temporary
storage writable. For Apptainer or Singularity deployments converted from an
OCI image:

- start the converted image with `/opt/grobid` as the working directory,
  because the OCI command uses a relative executable path
- bind a writable host directory to `/opt/grobid/grobid-home/tmp`, because
  GROBID creates temporary files during PDF processing and a SIF image is
  read-only

## Implementation

The implemented bibliography package is `src/tabulus/bibliography/`:

- `models.py`: normalized bibliography and entry models
- `grobid.py`: GROBID TEI parsing and deterministic DOI extraction from raw
  bibliography text
- `grobid_client.py`: HTTP client for GROBID `processReferences`
- `output.py`: `references/bibliography.json` writer
- `pipeline.py`: one-PDF extraction pipeline

Raw reference text is preserved. DOI extraction at this step is deterministic
only when a DOI already appears in the extracted bibliography text. Step 4
must not query Crossref or other metadata services.

## Boundary

Bibliography extraction is separate from:

- reference-table classification, which routes reconstructed structured tables
  as reference-like or non-reference-like
- reference matching, which combines selected reference-like table rows with
  `references/bibliography.json`
- Step 6 scholarly reference resolution, which validates paper-level
  identities in a separate artifact
- Step 7 resolved export, which writes separate downstream CSV outputs without
  mutating bibliography extraction evidence

Raw reconstruction prediction CSVs remain untouched.

The implementation is unit-tested and has been exercised against a live GROBID
service. Reconstruction or bibliography accuracy must be evaluated separately
against suitable gold-standard data.

## Examples

### TabulusBench

[TabulusBench](https://zenodo.org/records/20230340) is the benchmark dataset used for concrete tutorial examples. It currently contains 252 papers across five domains:

- `Biomedicine_And_Health`
- `Agriculture_Food_And_Environmental_Systems`
- `Computer_Science_AI_And_Data_Science`
- `Energy_Materials_Chemical_Sciences`
- `Engineering_Robotics_And_Built_Infrastructure`

Step 4 can run on the full 252-paper benchmark because every paper has an original PDF. A 52-paper subset also contains `bibliography/gold.json`; that subset is used for quantitative Step 4 evaluation and contains 6,947 numbered gold bibliography entries. The gold files are benchmark annotations only: Step 4 does not read them, and generated artifacts must not be written into `bibliography/`.

Use portable roots for the examples:

```bash
export TABULUSBENCH="/path/to/tabulusbench"
export TABULUS_WORK="/path/to/tabulus-work"
```

### 1. Extract the bibliography for one paper

Use the same P4 paper followed through the preceding tutorial steps:

- paper ID: `P4`
- domain: `Biomedicine_And_Health`
- subdomain: `clinical_research`
- PDF: `Biomedicine_And_Health/clinical_research/P4/P4.pdf`

Define the input PDF and generated artifact root:

```bash
P4_PDF="$TABULUSBENCH/Biomedicine_And_Health/clinical_research/P4/P4.pdf"
P4_OUT="$TABULUS_WORK/stage4/Biomedicine_And_Health/clinical_research/P4"
```

Check that GROBID is reachable before extraction:

```bash
curl http://localhost:8070/api/isalive
```

Run Step 4 on the original P4 PDF:

```bash
tabulus extract-bibliography \
  --pdf "$P4_PDF" \
  --out "$P4_OUT" \
  --grobid-url http://localhost:8070 \
  --timeout-seconds 300
```

The resulting artifact is:

```text
$TABULUS_WORK/stage4/
  Biomedicine_And_Health/
    clinical_research/
      P4/
        references/
          bibliography.json
```

Inspect the generated bibliography artifact with:

```bash
python -m json.tool \
  "$P4_OUT/references/bibliography.json"
```

Each entry records the one-based `index`, preserved `raw` citation, extraction `source`, DOI when already present in the extracted citation, and structured title, authors, year, venue, volume, issue, and pages when GROBID supplies them. Those structured fields are optional extraction evidence, not guaranteed metadata.

### 2. Extract bibliographies for all 252 TabulusBench papers

The Step 4 CLI processes one PDF per invocation. A full TabulusBench run therefore uses external orchestration around the existing command rather than a collection-level Tabulus CLI command.

This portable Python loop recursively discovers benchmark paper PDFs with the layout `<domain>/<subdomain>/<paper>/<paper>.pdf`, preserves that hierarchy under `$TABULUS_WORK/stage4/`, skips completed papers with a valid `references/bibliography.json`, and continues after per-paper failures:

```bash
python - <<'PY'
from pathlib import Path
import json
import os
import subprocess

DOMAINS = {
    "Biomedicine_And_Health",
    "Agriculture_Food_And_Environmental_Systems",
    "Computer_Science_AI_And_Data_Science",
    "Energy_Materials_Chemical_Sciences",
    "Engineering_Robotics_And_Built_Infrastructure",
}

bench = Path(os.environ["TABULUSBENCH"])
work = Path(os.environ["TABULUS_WORK"])
output_root = work / "stage4"

def valid_existing_artifact(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("entries"), list)

paper_pdfs = []
for pdf in bench.rglob("*.pdf"):
    rel = pdf.relative_to(bench)
    if len(rel.parts) != 4:
        continue
    domain, subdomain, paper, filename = rel.parts
    if domain not in DOMAINS:
        continue
    if filename != f"{paper}.pdf":
        continue
    paper_pdfs.append((rel, pdf, output_root / domain / subdomain / paper))

paper_pdfs.sort(key=lambda item: item[0].as_posix())
print(f"Discovered {len(paper_pdfs)} TabulusBench paper PDFs; expected 252.")

failures = []
for index, (rel, pdf, out_dir) in enumerate(paper_pdfs, start=1):
    artifact = out_dir / "references" / "bibliography.json"
    label = rel.as_posix()
    if valid_existing_artifact(artifact):
        print(f"[{index}/{len(paper_pdfs)}] skip existing {label}")
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{index}/{len(paper_pdfs)}] extract {label}")
    command = [
        "tabulus",
        "extract-bibliography",
        "--pdf",
        str(pdf),
        "--out",
        str(out_dir),
        "--grobid-url",
        "http://localhost:8070",
        "--timeout-seconds",
        "900",
    ]
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        failures.append(label)
        print(f"  failed with return code {result.returncode}")

print(f"Completed with {len(failures)} failed paper(s).")
for label in failures:
    print(f"FAILED {label}")
PY
```

The longer timeout is useful for very large bibliographies such as P251 and P252. The loop is sequential, writes only under `$TABULUS_WORK/stage4/`, and never writes into TabulusBench gold directories.

The generated layout is:

```text
$TABULUS_WORK/stage4/
  <domain>/
    <subdomain>/
      <paper>/
        references/
          bibliography.json
```

Representative outputs include:

```text
$TABULUS_WORK/stage4/Biomedicine_And_Health/clinical_research/P4/references/bibliography.json
$TABULUS_WORK/stage4/<domain>/<subdomain>/P251/references/bibliography.json
$TABULUS_WORK/stage4/<domain>/<subdomain>/P252/references/bibliography.json
```

After a complete run, the discovered paper-PDF count and generated bibliography-artifact count should both be 252:

```bash
python - <<'PY'
from pathlib import Path
import os

DOMAINS = {
    "Biomedicine_And_Health",
    "Agriculture_Food_And_Environmental_Systems",
    "Computer_Science_AI_And_Data_Science",
    "Energy_Materials_Chemical_Sciences",
    "Engineering_Robotics_And_Built_Infrastructure",
}

bench = Path(os.environ["TABULUSBENCH"])
work = Path(os.environ["TABULUS_WORK"])

paper_pdfs = []
for pdf in bench.rglob("*.pdf"):
    rel = pdf.relative_to(bench)
    if len(rel.parts) == 4 and rel.parts[0] in DOMAINS and rel.name == f"{rel.parts[2]}.pdf":
        paper_pdfs.append(pdf)

artifacts = list((work / "stage4").rglob("references/bibliography.json"))
print(f"TabulusBench paper PDFs: {len(paper_pdfs)}")
print(f"Generated Step 4 artifacts: {len(artifacts)}")
PY
```

Expected complete-run counts:

```text
TabulusBench paper PDFs: 252
Generated Step 4 artifacts: 252
```

### 3. Gold-bibliography subset for evaluation

A subset of 52 TabulusBench papers contains manually curated bibliography gold:

```text
bibliography/gold.json
```

Those 52 papers contain 6,947 numbered gold bibliography entries and are used for quantitative Step 4 evaluation. The remaining 200 papers can still be processed by Step 4, but they cannot contribute to gold-standard bibliography accuracy measurements. Each gold entry records the benchmark reference position `nr` and raw reference string `ref`; the generated extraction artifact remains `references/bibliography.json` under the Step 4 output root.

Count the gold bibliography files with:

```bash
find "$TABULUSBENCH" -path '*/bibliography/gold.json' | wc -l
```

Expected result:

```text
52
```

To sum the gold entries:

```bash
python - <<'PY'
from pathlib import Path
import json
import os

total = 0
for path in Path(os.environ["TABULUSBENCH"]).rglob("bibliography/gold.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        total += len(payload)
    elif isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        total += len(payload["entries"])
    else:
        raise ValueError(f"Unsupported gold bibliography format: {path}")
print(total)
PY
```

Expected result:

```text
6947
```

### 4. Continue to Step 5

Step 4 does not need Step 3 to run. The two branches meet at Step 5: Step 3 produces `selected_reference_tables.json` from reconstructed table outputs, and Step 4 produces `references/bibliography.json` from the original PDF. Step 5 then matches table reference tokens to bibliography positions without Step 4 modifying or consuming any table reconstruction files.
