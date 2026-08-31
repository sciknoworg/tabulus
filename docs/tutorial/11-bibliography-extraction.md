# Planned Stage: Bibliography Extraction

## Goal

Extract bibliography entries from the original PDF.

## Input

`input/paper.pdf` and `metadata/reference_section.json`.

## Output

`references/bibliography.json`.

## Module Contract

See `data-contracts/bibliography-json.md`.

## Default Implementation

This stage is retained in the legacy thesis workflow but is not yet implemented in the rebuilt `src/tabulus` library.

The target workflow uses GROBID first. If GROBID fails or returns unusable bibliography entries, a Kreuzberg OCR fallback can extract raw reference-section text and apply bibliography regex patterns. The resulting bibliography artifact should be evaluated independently from table reconstruction quality.

## Alternative Adapters

- GROBID
- Kreuzberg OCR plus regex
- PaddleOCR reference pages
- External metadata services

## Verification

The step succeeds when bibliography entries are emitted in a normalized list with `index`, `raw`, `doi`, and `source`.
