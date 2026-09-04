# Step 4: Bibliography Extraction

## Goal

Extract normalized bibliography entries from the original scientific PDF.

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

classified reference-like table + bibliography.json
      |
      v
reference matching -> DOI resolution -> resolved table export
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

See {doc}`../data-contracts/bibliography-json`.

## Python API

The implemented public interface processes one PDF at a time:

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

There is not yet a `tabulus` CLI subcommand or collection-level helper for
bibliography extraction. For a directory of PDFs, call the same API once per
PDF and choose a separate artifact root for each document:

```python
from pathlib import Path

from tabulus.bibliography.pipeline import extract_bibliography_artifact

input_dir = Path("INPUT_DIRECTORY")
output_dir = Path("OUTPUT_DIRECTORY")
grobid_url = "http://localhost:8070"

for pdf_path in sorted(input_dir.glob("*.pdf")):
    extract_bibliography_artifact(
        pdf_path,
        output_dir / pdf_path.stem,
        grobid_url=grobid_url,
    )
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
- `grobid.py`: GROBID TEI parsing and deterministic DOI extraction
- `grobid_client.py`: HTTP client for GROBID `processReferences`
- `output.py`: `references/bibliography.json` writer
- `pipeline.py`: one-PDF extraction pipeline

Raw reference text is preserved. DOI extraction at this stage is deterministic
only when a DOI is already present in the bibliography representation. Stage 4
must not query Crossref or other metadata services; missing DOI enrichment
belongs to the later DOI-resolution stage.

## Boundary

Bibliography extraction is separate from:

- reference-table classification, which routes reconstructed structured tables
  as reference-like or non-reference-like
- reference matching, which combines classified reference-like table rows with
  `references/bibliography.json`
- DOI resolution, which enriches matched references later
- resolved export, which writes separate downstream outputs

Raw reconstruction prediction CSVs must remain untouched.

The implementation is unit-tested and has been exercised against a live GROBID
service. Reconstruction or bibliography accuracy must be evaluated separately
against suitable gold-standard data.
