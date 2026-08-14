# Step 7: Bibliography Extraction

## Goal

Extract bibliography entries from the original PDF.

## Input

`input/paper.pdf` and `metadata/reference_section.json`.

## Output

`references/bibliography.json`.

## Module Contract

See `data-contracts/bibliography-json.md`.

## Default Implementation

The current pipeline uses GROBID first. If GROBID fails, it can call Kreuzberg for OCR text and then apply bibliography regex patterns.

## Alternative Adapters

- GROBID
- Kreuzberg OCR plus regex
- PaddleOCR reference pages
- External metadata services

## Verification

The step succeeds when bibliography entries are emitted in a normalized list with `index`, `raw`, `doi`, and `source`.
