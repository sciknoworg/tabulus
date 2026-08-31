:orphan:

# Concept: Scientific Table Normalization

This page describes a future scientific-normalization layer. It is retained for old links and design context, but it is not part of the current visible runnable tutorial sequence.

The rebuilt library does not currently expose a standalone table-normalization command. Current structural parsing happens inside {doc}`08-table-ocr`:

```text
adapter-native output
      |
      v
shared HTML/Markdown parsing
      |
      v
rectangular parsed representation
      |
      v
prediction CSV
```

## Current Implemented Parsing

The current `tabulus.table_ocr.parsing` layer reads adapter-native Markdown/HTML text, prefers HTML `<table>...</table>` elements when present, falls back to GitHub-style pipe-table Markdown only when no HTML table is found, and returns a rectangular row representation with:

- `rows`
- `n_rows`
- `n_cols`
- `source`

For HTML tables, `rowspan` and `colspan` are expanded into the common rectangular row matrix. The original merged-cell value is retained only at the upper-left grid position; other grid positions covered by the span become empty-string placeholders. This preserves column alignment for CSV-style reconstruction, but it is structural expansion only, not semantic fill-down.

Invalid or non-positive span values fall back safely to a span of one. Rowspans that extend past the available HTML rows are clipped rather than creating synthetic rows. Markdown fallback behavior is unchanged.

This behavior is adapter-neutral and is useful for any reconstruction model that emits HTML tables.

The parser also provides deterministic OTSL-to-HTML normalization for adapters
that emit OTSL before the shared HTML parser runs. The supported OTSL
structural tokens are `fcel`, `ecel`, `lcel`, `ucel`, `xcel`, and `nl`.
Generated rows are rectangularized to the width of the widest OTSL row. This
preserves the model's generated structure for CSV-style output; it is not
semantic cell-content correction, column-count guessing, or heuristic repair.

## Future Scientific Normalization

Future scientific table normalization should remain separate from raw reconstruction output. It may add stable row/cell identifiers, semantic cleanup, formula handling, section-row interpretation, candidate selection, or other domain-specific logic after reconstruction evidence has already been preserved.

The prediction CSV remains the reconstruction-quality artifact. Later reference matching and DOI enrichment should produce separate resolved CSV files without overwriting it.
