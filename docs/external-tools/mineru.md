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

A representative MinerU-native output directory looks like:

```text
<document>/
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
            └── MinerU-native document/method hierarchy
```

Tabulus chooses the profiler/backend root. MinerU creates its native document and method hierarchy beneath it. Do not flatten or rename MinerU-native files.
