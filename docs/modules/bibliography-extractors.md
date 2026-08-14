# Bibliography Extractors

Bibliography extractors produce normalized bibliography entries from a paper.

## Responsibility

- Extract bibliography entries.
- Preserve raw reference strings.
- Extract DOI values when present.
- Record source adapter and fallback status.

## Current Adapters

- GROBID TEI extraction
- Kreuzberg OCR plus regex fallback

The fallback currently accepts numbered bibliography output only.
