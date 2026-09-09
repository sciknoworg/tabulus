<p align="center">
  <img src="./assets/logo.png" alt="Tabulus logo" width="170"/>
</p>

# End-to-End Scientific Review Table Digitization Pipeline

<p align="center">
  <a href="https://tabulus.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/readthedocs/tabulus/latest?label=Read%20the%20Docs&logo=readthedocs" alt="Read the Docs"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.20741284">
    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20741284-blue.svg" alt="TabulusBench DOI"/>
  </a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python >= 3.10"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
</p>

Tabulus is a modular framework for digitizing scientific review tables into
structured, citation-aware data. It starts from scientific PDFs, reconstructs
reference-containing tables, extracts the paper bibliography, and resolves table
citations through inspectable CLI stages and filesystem artifacts.

The rebuilt library currently runs through **Stage 6 paper-level scholarly
reference resolution**. Stage 7 resolved CSV export and a single end-to-end
`tabulus run` command are planned.

## ✨ Highlights

- PDF profiling and table detection with MinerU.
- Canonical table crops for reproducible adapter comparisons.
- Table reconstruction through a registry of OCR, table-structure, and document
  vision-language-model adapters.
- Deterministic reference-table classification and table-cell-to-bibliography
  matching.
- GROBID-backed bibliography extraction from the original PDF.
- Conservative Stage 6 scholarly reference resolution using deterministic
  evidence, Crossref/CORE metadata, and bounded LLM adjudication.
- Native table-reconstruction evaluation with Relative Mapping Similarity
  (RMS).

## 🧭 Pipeline

```text
PDF
 ├─ Stage 1: profile PDF and export canonical table crops
 │    └─ Stage 2: reconstruct tables
 │         └─ Stage 3: classify reference-containing tables
 └─ Stage 4: extract bibliography from the original PDF

Stage 3 selected tables + Stage 4 bibliography
 └─ Stage 5: match table citations to bibliography positions
      └─ Stage 6: resolve paper-level scholarly identities
           └─ Stage 7: resolved CSV export (planned)
```

Stage 4 is a parallel PDF-level branch. Stage 6 resolves each linked
bibliography entry once per paper, not once per table cell or reconstruction
adapter.

## 🚀 Install

```bash
git clone https://github.com/sciknoworg/tabulus.git
cd tabulus
python -m pip install -e ".[dev]"
tabulus --help
```

For environment-specific setup, see the
[Windows/CPU](https://tabulus.readthedocs.io/en/latest/installation/windows-cpu.html),
[GPU server](https://tabulus.readthedocs.io/en/latest/installation/gpu-server.html),
and [Python library](https://tabulus.readthedocs.io/en/latest/installation/python-library.html)
installation guides.

## ⚡ Quick Start

```bash
PDF="/path/to/paper.pdf"
ARTIFACT_ROOT="/path/to/tabulus-artifacts/paper"
CROP_ROOT="/path/to/tabulus-output/table-crops/paper"
RECONSTRUCTION="$CROP_ROOT/reconstructions/tesseract-tatr"
```

```bash
# Stage 1: PDF profiling and canonical crop export
tabulus profile --pdf "$PDF" --backend pipeline --method auto

# Stage 2: table reconstruction
tabulus reconstruct-tables \
  --crops "$CROP_ROOT" \
  --adapter tesseract-tatr \
  --device cpu

# Stage 3: reference-table classification
tabulus classify-reference-tables \
  --reconstruction "$RECONSTRUCTION"

# Stage 4: bibliography extraction
tabulus extract-bibliography \
  --pdf "$PDF" \
  --out "$ARTIFACT_ROOT" \
  --grobid-url http://localhost:8070

# Stage 5: table-cell to bibliography-position matching
tabulus match-references \
  --selected "$RECONSTRUCTION/selected_reference_tables.json" \
  --bibliography "$ARTIFACT_ROOT/references/bibliography.json"

# Stage 6: paper-level scholarly reference resolution
tabulus resolve-references \
  --bibliography "$ARTIFACT_ROOT/references/bibliography.json" \
  --reference-matches "$RECONSTRUCTION/references/reference_matches.json" \
  --out "$ARTIFACT_ROOT"
```

Stage 6 requires Crossref, CORE, and OpenAI-compatible LLM configuration. The
full option list and credential handling are documented in the
[Stage 6 tutorial](https://tabulus.readthedocs.io/en/latest/tutorial/13-doi-resolution.html).

## 📦 Main Artifacts

| Stage | Main artifact |
| --- | --- |
| Stage 1 | `tables_index.json` and canonical crop images |
| Stage 2 | `native/`, `parsed/`, `predictions/`, `batch_summary.json` |
| Stage 3 | `reference_table_classification.json`, `selected_reference_tables.json` |
| Stage 4 | `references/bibliography.json` |
| Stage 5 | `references/reference_matches.json` |
| Stage 6 | `references/reference_resolution.json` |
| Stage 7 | resolved CSV export, planned |

See the
[Data Contracts](https://tabulus.readthedocs.io/en/latest/data-contracts/run-directory.html)
for exact schemas and filesystem layout.

## 🧪 TabulusBench

TabulusBench is the companion benchmark dataset used by the tutorials and
evaluation documentation. It contains scientific PDFs, annotated
reference-containing table crops, table gold CSVs, and bibliography gold for a
subset of papers.

- [TabulusBench on Zenodo](https://zenodo.org/records/20230340)

The documentation uses `P4` as the canonical one-paper worked example. Benchmark
gold material is read-only; Tabulus runs should write their generated artifacts
outside the gold directories.

## 🛠️ Development

```bash
python -m pip install -e ".[dev]"
pytest
git diff --check
```

The supported public library lives in `src/tabulus`. Retained legacy and
research directories are preserved for provenance, but they are not the current
public API unless the documentation says so explicitly.

## 📖 Documentation

- [ReadTheDocs](https://tabulus.readthedocs.io/)
- [Core pipeline overview](https://tabulus.readthedocs.io/en/latest/tutorial/00-overview.html)
- [Stage 2 table reconstruction](https://tabulus.readthedocs.io/en/latest/tutorial/08-table-ocr.html)
- [External tools](https://tabulus.readthedocs.io/en/latest/external-tools/mineru.html)
- [Evaluation overview](https://tabulus.readthedocs.io/en/latest/evaluation/overview.html)

## 📑 Citation

Citation information will be added after the associated thesis or software
release is published.

## 📜 License

Tabulus is released under the [MIT License](LICENSE).
