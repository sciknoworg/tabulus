# Current State

This page is an engineering snapshot of the rebuilt installable Tabulus
library in this repository. For normal usage, start with
{doc}`../tutorial/00-overview` and the installation page for your machine.

The current `src/tabulus` package implements the persisted reference-processing
pipeline through Step 7 resolved CSV export. Run-report/QA bundle generation,
standalone scientific table normalization, and complete `tabulus run`
orchestration are not implemented in this checkout.

## Runnable Steps

The rebuilt library currently exposes standalone command-line steps:

```bash
tabulus profile --pdf /path/to/paper.pdf --backend pipeline

tabulus reconstruct-tables \
  --crops /path/to/tabulus-output/table-crops/<paper> \
  --adapter paddleocr-vl \
  --device gpu:0

tabulus classify-reference-tables \
  --reconstruction /path/to/tabulus-output/table-crops/<paper>/reconstructions/paddleocr-vl

tabulus extract-bibliography \
  --pdf /path/to/paper.pdf \
  --out /path/to/artifact-root \
  --grobid-url http://localhost:8070

tabulus match-references \
  --selected /path/to/selected_reference_tables.json \
  --bibliography /path/to/artifact-root/references/bibliography.json

tabulus resolve-references \
  --bibliography /path/to/artifact-root/references/bibliography.json \
  --reference-matches /path/to/reconstruction/references/reference_matches.json \
  --out /path/to/artifact-root

tabulus export-resolved-csv \
  --reference-matches /path/to/reconstruction/references/reference_matches.json \
  --reference-resolution /path/to/artifact-root/references/reference_resolution.json
```

Each command writes a persisted artifact that the next step can inspect or
consume. The current CLI does not expose a complete `tabulus run`
orchestrator.

## Implemented

`tabulus.mineru`
: MinerU-backed PDF profiling, existing-output discovery, recursive
  `*_content_list.json` lookup, table-region extraction, page/provenance
  normalization, and automatic canonical table-crop export.

`tabulus export-table-crops`
: Regenerates the canonical crop handoff from an existing MinerU output
  directory without rerunning MinerU.

`tabulus.table_ocr`
: Step 2 table reconstruction over canonical MinerU crops. The package
  provides the adapter protocol, lazy registry, batch runner, shared
  HTML/Markdown parser, deterministic OTSL-to-HTML normalization, native and
  parsed artifact writing, prediction CSV export, and batch summary manifests.

`tabulus reconstruct-tables`
: Runs one registered reconstruction adapter over one or more canonical crop
  roots. It preserves table IDs, processes each canonical crop independently,
  reuses one adapter instance across the command, and writes `native/`,
  `parsed/`, `predictions/`, and `batch_summary.json`.

`tabulus.reference_tables`
: One deterministic regex/rule classifier applied independently to each
  reconstruction method's outputs. It identifies reference-containing
  reconstructed-table instances. It consumes reconstruction manifests and
  parsed artifacts, writes `reference_table_classification.json` and
  `selected_reference_tables.json`, and keeps independent classifications
  separate from continuation-inherited decisions.

`tabulus.bibliography`
: GROBID-backed bibliography extraction for one original scientific PDF. It is
  available through `tabulus extract-bibliography` and the Python API. It posts
  the PDF to GROBID `processReferences`, requests raw citations, disables
  GROBID citation consolidation, preserves raw reference text, extracts DOI
  strings only when already present in that extracted text, preserves optional
  structured fields, and writes `references/bibliography.json`.

Step 5 reference matching
: Deterministic linking of selected reference-like table cells to entries in
  `references/bibliography.json`. Matching preserves row-level provenance,
  unmatched tokens, ambiguous candidates, and skipped-table diagnostics in
  `references/reference_matches.json` without modifying reconstruction
  prediction CSVs.

Step 6 reference resolution
: Paper-level scholarly reference resolution through
  `tabulus resolve-references`. Step 6 consumes the union of Step 5-linked
  bibliography indices, deduplicates by bibliography index, retrieves Crossref
  and CORE candidates, uses bounded LLM adjudication when deterministic
  evidence is insufficient, applies a final deterministic admissibility gate,
  checkpoints completed references, and writes
  `references/reference_resolution.json` only after all targets complete.

Step 7 resolved CSV export
: Deterministic joining of Step 5 physical-row matches with the Step 6
  paper-level registry through `tabulus export-resolved-csv`. Step 7 performs
  no network lookup or scholarly re-resolution. It writes physical resolved
  CSVs by default and can optionally materialize safe continuation-aware merged
  CSVs from Step 1 continuation topology while retaining the physical outputs.

## Step 2 Adapter Set

The current registered crop-consuming adapters are maintained in
{doc}`../tutorial/08-table-ocr`. Adapter-specific model revisions, prompts,
runtime versions, and upstream resources are documented in the External Tools
section.

MinerU `table_body` remains a native reconstruction candidate produced during
profiling rather than a crop-consuming adapter.

The core reconstruction policy is unchanged across adapters:

- every adapter receives the same canonical MinerU crop
- no adapter returns to the source PDF to choose a different table region
- no adapter output is semantically repaired to improve apparent quality
- no continued-table merging happens during reconstruction
- prediction CSVs are written only for `ok` results with exactly one parsed
  structured table

## Validation Boundary

The automated test suite is designed so most tests do not require heavyweight
GPU model execution. It covers:

- MinerU output discovery and profiling command behavior
- canonical table-crop export
- adapter registry metadata and lazy loading
- mocked behavior for registered reconstruction adapters
- shared HTML/Markdown parsing and OTSL normalization
- batch reconstruction input handling and artifact writing
- reference-table classification heuristics and manifest writing
- GROBID TEI bibliography parsing, HTTP request construction, and bibliography
  artifact writing
- deterministic Step 5 matching behavior and artifact writing
- Step 6 target collection, Crossref/CORE candidate normalization,
  deterministic scoring, bounded LLM contract validation, failover,
  checkpointing, and final artifact writing
- Step 7 deterministic CSV enrichment, resolved table manifests, and optional
  conservative continuation merging

Real-model GPU validations are operational engineering checks. They confirm
that adapters can load, run through the Tabulus CLI, and produce the expected
artifact layers in validated environments. They are not reconstruction
accuracy, recall, precision, F1, or model-ranking evidence.

## Output Boundary

Current reconstruction outputs are pre-reference-resolution artifacts:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      native/
      parsed/
      predictions/
      batch_summary.json
      reference_table_classification.json
      selected_reference_tables.json
```

`native/` preserves adapter-native evidence. `parsed/` preserves the common
structured representation. `predictions/` contains raw reconstruction CSVs.
`reference_table_classification.json` and `selected_reference_tables.json` are
downstream routing/classification manifests and do not overwrite
reconstruction predictions.

Current bibliography extraction writes:

```text
<artifact-root>/
  references/
    bibliography.json
```

This artifact is produced from the original PDF, not from MinerU crops or
prediction CSV files.

Current reference matching writes by default:

```text
<reconstruction-directory>/
  references/
    reference_matches.json
```

This artifact is produced from selected reference-like tables and
`references/bibliography.json`.

Current Step 6 reference resolution writes:

```text
<artifact-root>/
  references/
    reference_resolution.json
```

A resumable in-progress run may also write
`references/reference_resolution.checkpoint.json` under the same artifact root.

Current Step 7 resolved export writes by default:

```text
<reconstruction-directory>/
  resolved_reference_tables/
    <prediction-stem>_resolved.csv
    resolved_tables.json
```

With `--merge-continuations`, successful logical continuation merges are
additional files under `resolved_reference_tables/merged/`; physical resolved
CSVs are always retained.

For the full filesystem contract, see {doc}`../data-contracts/run-directory`.

## Reference-Processing Architecture

Step 3 table selection and Step 4 bibliography extraction are parallel
branches from the paper. Step 5 matches table-cell references to bibliography
positions. Step 6 resolves the union of matched bibliography indices once at
paper scope:

```text
PAPER
  |
  +--> table branch
  |      |
  |      v
  |    Step 3 selected reference tables
  |      |
  |      v
  |    Step 5 reference_matches.json
  |
  +--> bibliography branch
         |
         v
       Step 4 bibliography.json

union of matched bibliography indices
  |
  v
Step 6 reference_resolution.json
  |
  v
Step 7 resolved_reference_tables/
```

Crossref, CORE, and LLM providers are used only in Step 6. Step 4 remains
GROBID extraction, and Step 5 remains deterministic offline matching.

Step 7 is deterministic export. It joins validated, rejected, and unmatched
reference evidence back to physical resolved CSV rows and can optionally
materialize compatible continuation groups without deleting physical outputs.

## Not Yet Rebuilt

The following remain planned or historical in the rebuilt library unless a
future implementation changes this page:

- run report / QA bundle generation
- full `tabulus run` orchestration
- standalone scientific table normalization command
- corpus-scale bibliography validation
- Kreuzberg fallback

Historical thesis code and older evaluation material may still mention some of
these systems. Those references should not be read as current runnable
features of `src/tabulus`.
