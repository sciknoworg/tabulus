# Run Directory

Each paper should have one run directory.

```text
runs/
  P51/
    input/
      paper.pdf
    metadata/
      pdf_profile.json
    pages/
    layout/
    tables/
      crops/
    references/
    resolved_reference_tables/
    report/
```

The run directory is the shared filesystem contract between modules.

Adapter-owned source outputs should be retained inside the run directory instead of being treated as the normalized contract. For the current MinerU adapter, keep the full MinerU output directory, including `content_list.json`, `layout.pdf`, `middle.json`, `model.json`, reconstructed Markdown, and generated images.

Downstream modules should consume the normalized Tabulus outputs, such as `metadata/pdf_profile.json` and indexed table images, while debugging and evaluation can trace those outputs back to the original MinerU files.
