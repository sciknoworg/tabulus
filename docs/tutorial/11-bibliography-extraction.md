# Planned Stage: Bibliography Extraction

## Goal

Extract normalized bibliography entries from the original scientific PDF.

This is a PDF-level branch. It runs in parallel with the table-processing
branch and does not consume MinerU table crops, reconstructed prediction CSVs,
or reference-table classification output.

```text
Original scientific PDF
      |
      +--> MinerU table detection
      |         |
      |         v
      |   canonical table crops
      |         |
      |         v
      |   table reconstruction adapters
      |         |
      |         v
      |   structured table representations
      |         |
      |         v
      |   reference-table classification
      |
      +--> GROBID bibliography extraction
                |
                v
          references/bibliography.json

reference-table classification + bibliography.json
      |
      v
reference matching -> DOI resolution / enrichment -> resolved scientific table export
```

## Input

The input is the original scientific PDF.

## Output

The normalized handoff is:

```text
references/
  bibliography.json
```

See {doc}`../data-contracts/bibliography-json`.

## Extractor Contract

GROBID is the intended primary bibliography extractor. The bibliography branch
should preserve raw bibliography strings, normalize entries into
`references/bibliography.json`, and extract DOI values deterministically when a
DOI is already present in the bibliography text.

Stage 4 must not call Crossref or perform external DOI resolution. Missing DOI
values are handled later by the DOI-resolution branch after reference matching.

Fallback bibliography extraction is planned as an optional recovery path, not
part of the table-reconstruction branch.

## Boundary

Bibliography extraction should leave raw reconstruction prediction CSVs
untouched. The table-processing branch and bibliography branch converge only
at reference matching:

```text
classified reference-like table
  +
references/bibliography.json
  |
  v
reference matching
```

Live GROBID service integration and corpus-scale validation are separate
deployment and evaluation steps.
