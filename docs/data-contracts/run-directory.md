# Run Directory

The long-term normalized Tabulus pipeline should give each paper one run directory. This is the future shared filesystem contract between modules:

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

Adapter-owned source outputs should be retained inside the run directory instead of being treated as the normalized contract. For the current MinerU adapter, keep the full MinerU output directory, including `content_list.json`, `layout.pdf`, `middle.json`, `model.json`, reconstructed Markdown, and generated images.

Downstream modules should consume the normalized Tabulus outputs, such as `metadata/pdf_profile.json` and indexed table images, while debugging and evaluation can trace those outputs back to the original MinerU files.

## Current Profiling Output Convention

The current implemented `tabulus profile` command uses a simpler profiling-output convention when `--out` is omitted:

```text
<PDF directory>/
  tabulus-output/
    <profiler>/
      <resolved-backend>/
    table-crops/
      <PDF stem>/
```

For the current MinerU CPU path:

```text
<PDF directory>/
  tabulus-output/
    mineru/
      pipeline/
    table-crops/
      <PDF stem>/
```

`mineru` is the profiler. `pipeline` and `hybrid-engine` are MinerU backends.

If `hybrid-engine` is requested but Tabulus falls back to `pipeline`, the automatic output directory uses the resolved backend name:

```text
tabulus-output/mineru/pipeline/
```

`--out` remains available for explicit profiling-root override. `--table-crops-out` separately overrides the normalized table-crop handoff directory, and `--no-export-table-crops` disables automatic crop export.

MinerU retains its own native output hierarchy under the profiler/backend root, typically:

```text
tabulus-output/
  mineru/
    <resolved-backend>/
      <document>/
        <MinerU-native run directory>/
          images/
          <document>_content_list.json
          <document>_content_list_v2.json
          <document>_layout.pdf
          <document>_middle.json
          <document>_model.json
          <document>_origin.pdf
          <document>.md
```

Do not flatten or rename MinerU-native output files. The current Tabulus reader recursively finds `*_content_list.json` and resolves table images from MinerU's `img_path` values. The normalized table-crop handoff copies those canonical MinerU crops into:

```text
tabulus-output/
  table-crops/
    <PDF stem>/
      tables_index.json
      images/
```
