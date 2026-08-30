# MinerU

MinerU is an external PDF parsing/profiling tool used by Tabulus.

Tabulus deliberately exposes only the MinerU functionality required by the current PDF-profiling workflow. Tabulus currently supports MinerU's `pipeline` and `hybrid-engine` backends, tested with MinerU 3.4.5.

Users who need MinerU features outside the Tabulus profiling workflow should consult the official [MinerU documentation](https://opendatalab.github.io/MinerU/) and [MinerU project repository](https://github.com/opendatalab/MinerU).

## MinerU Options Used By Tabulus

Tabulus invokes MinerU through `tabulus profile`.

```bash
tabulus profile --pdf "<paper.pdf>" --backend pipeline
tabulus profile --pdf "<paper.pdf>" --backend hybrid-engine
```

After a successful MinerU run, `tabulus profile` exports canonical MinerU table crops into the normalized Tabulus table-crop handoff by default. Use `--table-crops-out PATH` to override that handoff directory, or `--no-export-table-crops` to keep only the MinerU-native profiling output.

### Backend

`--backend pipeline`
: CPU-compatible MinerU backend used by the validated Windows/CPU workflow.

`--backend hybrid-engine`
: GPU-backed MinerU workflow used by the validated Linux GPU workflow. Tabulus checks GPU suitability before execution and can fall back to `pipeline` if the requirements are not satisfied.

### Method

`--method` selects the MinerU parsing mode exposed through the Tabulus CLI:

```text
             --method auto
                   |
         MinerU examines document
              +----+----+
              |         |
              v         v
        text extraction  OCR

--method txt  --> force text extraction
--method ocr  --> force OCR
```

`auto`
: Let MinerU determine whether native PDF text extraction or OCR should be used.

`txt`
: Force native PDF text extraction.

`ocr`
: Force OCR.

These are MinerU parsing modes, not Tabulus-specific OCR implementations.

### Effort

`--effort medium`
`--effort high`
: Processing effort for MinerU `hybrid-engine`.

`effort` controls the hybrid parsing effort / accuracy-versus-processing trade-off. Tabulus currently defaults to:

```bash
--effort high
```

Tabulus passes `--effort` only when the resolved MinerU backend is `hybrid-engine`.

## Fixed MinerU Settings

Tabulus currently fixes these MinerU settings internally:

```text
table=True
formula=False
image_analysis=False
```

This means:

- table extraction is enabled
- formula extraction is disabled
- image analysis is disabled

These settings are controlled by Tabulus in `src/tabulus/mineru/runner.py` and are not currently exposed as Tabulus CLI arguments.

## MinerU Output

A representative MinerU-native output tree looks like:

```text
<document>/
└── <MinerU-native run directory>/
    ├── images/
    ├── <document>_content_list.json
    ├── <document>_content_list_v2.json
    ├── <document>_layout.pdf
    ├── <document>_middle.json
    ├── <document>_model.json
    ├── <document>_origin.pdf
    └── <document>.md
```

`images/`
: MinerU-generated image assets, including table images referenced by structured output.

`<document>.md`
: Human-readable reconstructed Markdown representation.

`<document>_content_list.json`
: Flat structured content list currently used by the Tabulus table-discovery workflow. This is the most important MinerU output for the current Tabulus implementation.

`<document>_content_list_v2.json`
: Newer structured representation produced by MinerU.

`<document>_layout.pdf`
: Layout/debugging PDF useful for visually inspecting detected regions.

`<document>_middle.json`
: Detailed intermediate parsing representation useful for debugging.

`<document>_model.json`
: Lower-level/model inference output primarily useful for debugging.

`<document>_origin.pdf`
: MinerU's copy of the original PDF.

For details about how Tabulus consumes MinerU outputs downstream, see {doc}`../data-contracts/mineru-output-files`.

## Output Location In Tabulus

When `--out` is omitted, Tabulus chooses the profiler/backend output root:

```text
<PDF parent>/
└── tabulus-output/
    └── mineru/
        └── <resolved-backend>/
            └── <document>/
                └── <MinerU-native run directory>/
                    ├── ...
```

Tabulus owns only the profiling output root:

```text
<PDF parent>/tabulus-output/<profiler>/<resolved-backend>/
```

MinerU controls the document/run hierarchy below that root. Tabulus must not construct, flatten, rename, or assume the native run directory solely from `--method`; the native run-directory name is MinerU-owned behavior and can differ by backend, mode, and MinerU version.

After successful MinerU execution, Tabulus discovers the actual MinerU-native run directory from the generated `*_content_list.json`. Downstream table discovery also locates `*_content_list.json` recursively, so downstream consumers should target the actual MinerU-native run directory rather than hard-coding a directory name.

On success, Tabulus writes diagnostic logs beside the successful MinerU output inside the discovered native run directory:

```text
mineru_stdout.log
mineru_stderr.log
tabulus_run.txt
```

If MinerU fails before a native run directory can be identified, diagnostics may be written at the document level instead.

## Canonical Table Crops

MinerU is the canonical table-localization and crop-generation stage in the current clean Tabulus workflow. The normalized crop handoff is separate from MinerU's native output:

```text
<PDF parent>/
└── tabulus-output/
    └── table-crops/
        └── <document>/
            ├── tables_index.json
            └── images/
```

The standalone `tabulus export-table-crops` command remains useful when an expensive MinerU run should be reused, when the normalized handoff should be regenerated without rerunning MinerU, or when native MinerU output should remain untouched by Tabulus-specific downstream artifacts.

MinerU `table_body` is retained as a native reconstruction candidate produced during profiling. The crop-consuming reconstruction adapters, such as PaddleOCR-VL, Chandra OCR 2, and NuExtract3, operate later on the canonical crop images rather than on the original PDF.

For those crop-consuming integrations, see:

- {doc}`paddleocr-vl`
- {doc}`chandra`
- {doc}`nuextract3`

Validated examples:

Windows CPU, MinerU 3.4.5, `pipeline` + `auto`:

```text
tabulus-output/
└── mineru/
    └── pipeline/
        └── <document>/
            └── auto/
                └── ...
```

Linux GPU, MinerU 3.4.5, `hybrid-engine` + `auto` + `high`:

```text
tabulus-output/
└── mineru/
    └── hybrid-engine/
        └── <document>/
            └── hybrid_auto/
                └── ...
```

`hybrid_auto` is MinerU's native directory name for the validated `hybrid-engine` + `auto` run. It is not a Tabulus-created directory and should not be treated as a guarantee for future MinerU versions.
