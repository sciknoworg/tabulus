# MinerU

MinerU is the external PDF profiling tool used by the current Tabulus Step 1
workflow. Tabulus invokes MinerU, discovers its structured output, and exports
the canonical table-crop handoff used by Stage 2 reconstruction.

For the guided profiling workflow and CLI examples, see
{doc}`../tutorial/01-pdf-profiling`.

## Official Resources

- [MinerU documentation](https://opendatalab.github.io/MinerU/)
- [MinerU project repository](https://github.com/opendatalab/MinerU)

## Role In Tabulus

Tabulus currently supports MinerU as the only PDF profiler:

```bash
tabulus profile --profiler mineru ...
```

MinerU performs document/layout processing, table localization, and native
table extraction. Tabulus then:

- locates MinerU `*_content_list.json` files
- resolves MinerU table image paths
- normalizes page numbers, bounding boxes, captions, footnotes, and provenance
- retains MinerU `table_body` as a native reconstruction candidate
- exports canonical table crops and `tables_index.json`

The stable handoff for later Tabulus stages is:

```text
tabulus-output/
  table-crops/
    <paper>/
      tables_index.json
      images/
```

Stage 2 reconstruction adapters should consume this handoff rather than the
complete MinerU-native directory.

## MinerU Options Exposed By Tabulus

`tabulus profile` exposes a small MinerU-specific surface:

| Tabulus option | MinerU meaning |
| --- | --- |
| `--backend pipeline` | CPU-compatible MinerU backend. |
| `--backend hybrid-engine` | GPU-backed MinerU backend. |
| `--method auto` | Let MinerU choose text extraction or OCR handling. |
| `--method txt` | Ask MinerU to use native PDF text extraction. |
| `--method ocr` | Ask MinerU to use OCR. |
| `--effort medium` / `--effort high` | Processing effort for `hybrid-engine`. |

If `hybrid-engine` is requested but GPU requirements are not satisfied,
Tabulus reports the reason and falls back to `pipeline`. Automatic output
paths use the resolved backend name.

Tabulus currently fixes these MinerU settings internally:

```text
table=True
formula=False
image_analysis=False
```

They are not currently exposed as Tabulus CLI flags.

## Native Output

Tabulus owns only the profiling output root:

```text
<PDF parent>/tabulus-output/mineru/<resolved-backend>/
```

MinerU owns the document/run hierarchy beneath that root:

```text
tabulus-output/
  mineru/
    <resolved-backend>/
      <paper>/
        <MinerU-native run directory>/
          images/
          <paper>_content_list.json
          <paper>_content_list_v2.json
          <paper>_layout.pdf
          <paper>_middle.json
          <paper>_model.json
          <paper>_origin.pdf
          <paper>.md
```

`images/`
: MinerU-generated image assets, including table images referenced by
  structured output.

`<paper>_content_list.json`
: Flat structured content list currently used by the Tabulus table-discovery
  workflow.

`<paper>_content_list_v2.json`
: Newer structured representation produced by MinerU.

`<paper>_layout.pdf`
: Layout/debugging PDF useful for visually inspecting detected regions.

`<paper>_middle.json`
: Detailed intermediate parsing representation useful for debugging.

`<paper>_model.json`
: Lower-level/model inference output primarily useful for debugging.

`<paper>_origin.pdf`
: MinerU's copy of the original PDF.

`<paper>.md`
: MinerU's human-readable reconstructed Markdown representation.

For the detailed file contract, see
{doc}`../data-contracts/mineru-output-files`.

## Native Run Directory Discovery

After successful MinerU execution, Tabulus discovers the actual
`<MinerU-native run directory>` from the generated `*_content_list.json`.
Tabulus must not construct, flatten, rename, or assume that native run
directory solely from `--method`.

Validated MinerU 3.4.5 examples:

```text
pipeline/<paper>/auto/
hybrid-engine/<paper>/hybrid_auto/
```

These are observed MinerU-owned directory names from tested configurations,
not universal Tabulus naming rules.

On success, Tabulus writes diagnostic files beside the discovered MinerU
output:

```text
mineru_stdout.log
mineru_stderr.log
tabulus_run.txt
```

If MinerU fails before a native run directory can be identified, diagnostics
may be written at the document level instead.

## Boundary

MinerU is the profiling and canonical crop-generation stage in the current
workflow. It is separate from crop-consuming Stage 2 reconstruction adapters,
reference-table classification, bibliography extraction, reference matching,
DOI resolution, and final resolved CSV export.
