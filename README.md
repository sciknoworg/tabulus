<p align="center">
  <img src="./assets/logo.png" alt="Tabulus logo" width="170"/>
</p>

# 📚 Tabulus: Scientific PDF Table Extraction Pipeline

<p align="center">
  <a href="https://tabulus.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/readthedocs/tabulus/latest?label=Read%20the%20Docs&logo=readthedocs" alt="Read the Docs"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.20741284">
    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20741284-blue.svg" alt="Tabulus Bench DOI"/>
  </a>
</p>

## 🔍 Overview
Tabulus is a modular multi-stage pipeline for extracting structured table data from scientific PDF documents.

The system combines document analysis, OCR, bibliography extraction, reference matching, and DOI enrichment into a unified workflow that transforms scientific publications into machine-readable data suitable for further analysis, knowledge graph integration, and research evaluation.

The project was developed as part of a Master's thesis investigating scientific table extraction, OCR benchmarking, bibliography-aware processing, and structured scholarly knowledge extraction.

---

## ✨ Features

### 📄 Scientific Table Extraction

* Automated table detection from scientific PDFs
* Table cropping and preprocessing
* OCR-based table reconstruction
* Structured CSV generation

### 🔗 Bibliography-Aware Processing
* Reference-table classification for reconstructed tables
* Preserved separation between reconstruction predictions and reference routing
* Planned bibliography extraction from full publications
* Planned reference matching and DOI enrichment

### 📊 Research & Evaluation
* OCR benchmarking framework
* RMS-based table similarity evaluation
* Precision, Recall, and F1-score analysis
* Runtime benchmarking
* Reproducible evaluation workflows

### 🏗️ System Design
* Modular CLI and library architecture
* Explicit filesystem contracts between stages
* Separate ML environments for heavyweight adapters
* CPU and GPU reconstruction-adapter support
* Legacy service implementation retained separately from the rebuilt library

---

## ⚙️ Pipeline Workflow
```text
Scientific PDF
      |
      v
tabulus profile / MinerU
      |
      +--> MinerU table_body
      |
      +--> canonical MinerU table crops
                |
                +--> PaddleOCR-VL
                +--> Chandra OCR 2
                +--> NuExtract3
                +--> Tesseract + Table Transformer
                +--> RapidOCR + Docling TableFormer
                +--> Granite Vision 4.1 4B
                |
                v
      tabulus reconstruct-tables
                |
                v
          prediction CSVs
                |
                v
      tabulus classify-reference-tables
                |
                v
      planned: bibliography extraction,
      reference matching, DOI resolution,
      resolved CSV export, and run reporting
```

---

## 📁 Repository Structure
```text
tabulus/
│
├── assets/
│   ├── img/
│   └── logo.png
│
├── dataset/
│   └── README.md
│
├── evaluation/
│   ├── deplot/
│   ├── new_results/
│   ├── plots/
│   │   ├── reference_extraction/
│   │   ├── scripts/
│   │   └── table_extraction/
│   ├── scripts/
│   └── README.md
│
├── docs/
│   └── ...
│
├── src/
│   ├── tabulus/
│   │   ├── mineru/
│   │   ├── reference_tables/
│   │   ├── table_ocr/
│   │   └── cli.py
│   │
│   ├── legacy_tabulus/
│   │   └── ...
│   │
│   ├── ocr_models/
│   │   └── ...
│   │
│   └── README.md
│
├── tests/
│   └── ...
│
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 🧩 Main Components
| Component        | Purpose                                                    |
| ---------------- | ---------------------------------------------------------- |
| `src/tabulus`    | Current installable Tabulus library and CLI                |
| `src/legacy_tabulus` | Retained legacy thesis implementation                  |
| `src/ocr_models` | Historical OCR services, runners, and benchmarking components |
| `docs`           | ReadTheDocs documentation                                  |
| `tests`          | Current library test suite                                 |
| `evaluation`     | Evaluation scripts, metrics, and visualizations            |
| `dataset`        | Benchmark dataset documentation and ground-truth structure |
| `assets`         | Images and visual resources used in the documentation      |

Detailed documentation for each component is available in the corresponding README files.

---

## 🤖 OCR Technologies
The rebuilt Tabulus library currently integrates several OCR and document understanding approaches, with additional candidates retained for evaluation:

* MinerU
* PaddleOCR-VL
* Chandra OCR
* NuExtract3
* Tesseract + Table Transformer
* RapidOCR + Docling TableFormer
* Granite Vision 4.1 4B
* DeepSeek OCR 2 (future candidate)
* GROBID (legacy/reference-processing context)

---

## 🗄️ Dataset
The project uses a manually curated evaluation dataset containing:

* scientific publications,
* annotated tables,
* bibliography references,
* OCR outputs,
* DOI matching results,
* evaluation metrics.

The complete dataset exceeds 700 MB and is distributed separately.

See:

```text
dataset/README.md
```

for details.

---

## 📈 Evaluation
A comprehensive evaluation framework is included for analyzing:

* table extraction quality,
* OCR robustness,
* bibliography extraction performance,
* reference matching accuracy,
* DOI enrichment quality,
* runtime efficiency.

Generated benchmark plots and visualizations are available in:

```text
evaluation/plots/
```

See:

```text
evaluation/README.md
```

for detailed documentation.

---

## 🚀 Running the Current Rebuilt Workflow
Install the current library from the repository checkout:

```bash
python -m pip install -e ".[dev]"
```

The currently implemented stages are exposed as CLI commands:

```bash
tabulus profile --pdf /path/to/paper.pdf --backend pipeline

tabulus reconstruct-tables \
  --crops /path/to/tabulus-output/table-crops/<paper> \
  --adapter paddleocr-vl \
  --device gpu:0

tabulus classify-reference-tables \
  --reconstruction /path/to/tabulus-output/table-crops/<paper>/reconstructions/paddleocr-vl
```

See the ReadTheDocs installation pages for Windows CPU setup, GPU-server setup,
and adapter-specific environments. The legacy Docker/service workflow is not
the current rebuilt-library entry point.

---

## 📖 Documentation
Additional documentation is available in:

```text
https://tabulus.readthedocs.io/
docs/
evaluation/
dataset/
```

Each README contains detailed setup instructions, implementation details, API documentation, evaluation procedures, and usage examples.

---

## 🎓 Research Context
This repository accompanies a Master's thesis focused on:

* scientific table extraction,
* OCR benchmarking,
* bibliography-aware table processing,
* DOI enrichment,
* structured scientific knowledge extraction,
* reproducible research workflows.

---

## 📑 Citation
If you use this repository in your research, please cite the associated Master's thesis.

Citation information will be added after publication.

---

## 📜 License
This project is provided for research and educational purposes.
