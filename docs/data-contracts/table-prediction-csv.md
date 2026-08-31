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

Prediction CSV files may come from implemented and future reconstruction candidates:

- MinerU `table_body`
- MinerU crop -> PaddleOCR-VL
- MinerU crop -> Chandra
- MinerU crop -> NuExtract3
- MinerU crop -> Tesseract + Table Transformer
- MinerU crop -> RapidOCR + Docling TableFormer
- MinerU crop -> Granite Vision 4.1 4B
- MinerU crop -> TRivia-3B
- MinerU crop -> GLM-OCR
- MinerU crop -> Dolphin-v2
- MinerU crop -> DeepSeek-OCR-2

All candidates should be exported through the same normalized CSV shape before scoring. Adapter-native JSON, Markdown, HTML, or OCR text should be preserved separately for provenance and debugging.

The implemented `tabulus reconstruct-tables` command writes prediction CSV files under:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      predictions/
```

A prediction CSV is written only when the table reconstruction result has status `ok` and exactly one parsed table is available for that crop. If the adapter returns an error, no table, or multiple parsed tables from one canonical crop, Tabulus preserves the native and parsed artifacts plus a warning or error instead of choosing an arbitrary CSV.

Filename stems preserve the physical crop identity. For example:

```text
page_006_table_001.csv
```

means page 6 and Tabulus physical `table_id` 1. The `table_id` is derived from the MinerU discovery sequence and is not necessarily the table number printed in the scientific article. Each physical MinerU crop remains independent through reconstruction; continued-table merging is not currently performed at this stage.

Do not confuse prediction CSV files with {doc}`resolved-csv`. A resolved CSV is a later user-facing artifact for a reference-containing table after bibliography matching and DOI resolution.
