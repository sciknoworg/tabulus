# PDF Profilers

PDF profilers analyze input papers and emit table images plus structured metadata.

In the clean Tabulus workflow, PDF profiling is the first major digitization module. It is not a separate upload step and not only a metadata check. It is the module that reads the scientific PDF and produces the table images consumed by table reconstruction.

## Responsibility

- Validate that the input exists and is a PDF.
- Create the run directory if needed.
- Copy or link the source PDF if the run layout requires it.
- Perform page/layout analysis.
- Detect tables.
- Determine table bounding boxes.
- Save table crop images.
- Capture captions and footnotes.
- Write structured JSON.

## Default Tooling

Use Python standard library tools:

- `pathlib`
- `shutil`
- `hashlib`

The current adapter is MinerU.

The current Python script is:

```text
src/Tabulus/mineru_service/app/table_extraction_benchmark/runners/mineru_tables_png_runner.py
```

It handles MinerU outputs by reading `content_list.json`, filtering table entries, copying each generated `img_path`, and writing `tables_index.json`.

It does not crop the PDF from bounding boxes itself. MinerU has already generated the crop image.

## Adapter Ideas

- MinerU
- Docling or another document layout parser
- Custom table detector plus crop exporter
- Future web-upload adapter that produces the same file contract
