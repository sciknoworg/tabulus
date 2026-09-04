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

## Planned Adapters

GROBID TEI extraction is the intended primary extractor.

An OCR/regex fallback remains an optional planned recovery path. It should not
be treated as part of table reconstruction.

Bibliography extraction must not call Crossref or perform external DOI
resolution. External DOI lookup belongs to the later DOI-resolution stage after
reference matching.
