# Table OCR Adapters

Table OCR adapters convert table crop images into structured rows.

## Responsibility

- Accept one or more table crop images.
- Return rows, columns, raw model output, and parse status.
- Never silently drop failed tables.

## Current Adapter

The new library does not yet implement a table OCR adapter. The target adapter is PaddleOCR-VL.

## Other Experiment Adapters

The repository also contains runners or components for DeepSeek OCR, Chandra, Kreuzberg, and NuExtract3. These belong behind the same table OCR contract.
