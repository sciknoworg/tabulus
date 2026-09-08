# Step 4: Bibliography Extraction

## Goal

Extract an ordered bibliography from the original scientific PDF using GROBID.
Stage 4 is extraction only; it performs no Crossref, CORE, LLM, or
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
Stage 5 reference matching
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
Stage 4.

## Command Line

The normal Stage 4 command is:

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
outside Stage 4.

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
consolidation so external metadata lookup remains outside Stage 4.

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

Raw reference text is preserved. DOI extraction at this stage is deterministic
only when a DOI already appears in the extracted bibliography text. Stage 4
must not query Crossref or other metadata services.

## Boundary

Bibliography extraction is separate from:

- reference-table classification, which routes reconstructed structured tables
  as reference-like or non-reference-like
- reference matching, which combines selected reference-like table rows with
  `references/bibliography.json`
- Stage 6 scholarly reference resolution, which validates paper-level
  identities in a separate artifact
- planned Stage 7 resolved export, which would write separate downstream outputs

Raw reconstruction prediction CSVs remain untouched.

The implementation is unit-tested and has been exercised against a live GROBID
service. Reconstruction or bibliography accuracy must be evaluated separately
against suitable gold-standard data.
