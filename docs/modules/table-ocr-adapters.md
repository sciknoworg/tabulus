# Table OCR Adapters

Table OCR and Structure Extraction adapters convert normalized table crop images into structured tables.

In the current clean Tabulus workflow, MinerU is the canonical table-localization and crop-generation stage. Prior experimental work selected MinerU for this role because it was the strongest and most efficient option for locating scientific tables and producing useful crop images. The table OCR adapter layer is extensible, but its contract starts from the MinerU-generated crop rather than the original PDF.

## Responsibility

- Accept one or more normalized table crop images.
- Receive and retain the Tabulus table identifier and provenance.
- Reconstruct table structure and content.
- Return a structured result.
- Preserve adapter-native output where useful.
- Explicitly record empty or error results.
- Never silently drop failed tables.

Conceptually:

```text
MinerU table crop
       |
       v
Table reconstruction adapter
       |
       v
structured table
```

not:

```text
original PDF
       |
       v
each OCR adapter independently detects and crops tables
```

The adapter should focus on cell text, rows, columns, table structure, and adapter-native structured output while preserving the table ID and MinerU provenance supplied by the normalized Tabulus handoff.

Adapter-native outputs may differ. A table-reconstruction adapter may produce HTML, Markdown, CSV-like structures, JSON, or model-specific structured output. Downstream Tabulus code should normalize those outputs into the same common table representation before evaluation or export.

## Current Adapter

The new library does not yet implement a table OCR adapter. PaddleOCR-VL is the first/default adapter to be implemented in the clean library.

## Other Experiment Adapters

Other experiment adapters already represented in the repository include:

- DeepSeek OCR
- Chandra
- Kreuzberg
- NuExtract3

These belong behind the same Table OCR and Structure Extraction contract. They are alternative table-reconstruction adapters, not stages that run after PaddleOCR-VL.

MinerU `table_body` is slightly different: it is produced during PDF profiling rather than by consuming the normalized table-crop handoff. It should still satisfy the same downstream comparison and normalization concept as another candidate table reconstruction.
