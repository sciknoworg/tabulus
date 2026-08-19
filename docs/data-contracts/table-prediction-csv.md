# Table Prediction CSV

A table prediction CSV is the reconstructed table before reference resolution or DOI enrichment.

It is the artifact compared against a manually curated ground-truth CSV during table-reconstruction evaluation. It must not be overwritten by later enrichment stages.

```text
normalized reconstruction
        |
        v
prediction CSV
        |
        v
RMS / DePlot evaluation against ground-truth CSV
```

Prediction CSV files may come from different reconstruction candidates:

- MinerU `table_body`
- MinerU crop -> PaddleOCR-VL
- MinerU crop -> DeepSeek OCR
- MinerU crop -> Chandra
- MinerU crop -> Kreuzberg
- MinerU crop -> NuExtract3

All candidates should be exported through the same normalized CSV shape before scoring. Adapter-native JSON, Markdown, HTML, or OCR text should be preserved separately for provenance and debugging.

The implemented `tabulus reconstruct-tables` command writes prediction CSV files under:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      predictions/
```

A prediction CSV is written only when the table OCR result has status `ok` and exactly one parsed table is available for that crop. If the adapter returns an error, no table, or multiple parsed tables from one canonical crop, Tabulus preserves the native and parsed artifacts plus a warning or error instead of choosing an arbitrary CSV.

Do not confuse prediction CSV files with {doc}`resolved-csv`. A resolved CSV is a later user-facing artifact for a reference-containing table after bibliography matching and DOI resolution.
