# Bibliography Extractors

Bibliography extractors produce normalized bibliography entries from a paper.
They are a PDF-level branch: they read the original scientific PDF and do not
consume MinerU table crops, reconstruction prediction CSVs, or
reference-table classification output.

## Responsibility

- Extract bibliography entries.
- Preserve raw reference strings.
- Extract DOI values deterministically when they are already present in the
  bibliography text.
- Record source adapter and fallback status.
- Write `references/bibliography.json`.
- Leave reconstruction prediction CSVs untouched.

## Implemented Adapter

GROBID TEI extraction is implemented in `src/tabulus/bibliography/`.

The current implementation:

- sends the original PDF to a GROBID service over HTTP
- calls GROBID `processReferences`
- requests raw citations
- disables GROBID citation consolidation
- parses the returned TEI into normalized bibliography entries
- writes `references/bibliography.json`

The implemented modules are:

- `models.py`
- `grobid.py`
- `grobid_client.py`
- `output.py`
- `pipeline.py`

Bibliography extraction must not call Crossref or perform external DOI
resolution. External DOI lookup belongs to the later DOI-resolution stage after
reference matching.

## Planned Fallback

An OCR/regex fallback remains an optional planned recovery path. It should not
be treated as part of table reconstruction.
