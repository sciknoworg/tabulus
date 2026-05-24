<p align="center">
  <img src="./assets/logo.png" alt="Tabulus logo" width="150"/>
</p>

# Scientific PDF Table Extraction Pipeline

A modular multi-stage pipeline for extracting scientific tables, detecting bibliography references, resolving DOI information, and generating structured machine-readable outputs from scientific PDF documents.

---

# Overview

This project was developed as part of a Master's thesis focusing on:

- scientific table extraction,
- OCR benchmarking,
- bibliography-aware table processing,
- DOI enrichment,
- structured scientific knowledge extraction.

The pipeline combines multiple OCR and document-processing technologies into a unified workflow capable of processing complete scientific publications automatically.

---

# Main Features

- Scientific PDF processing
- Automated table detection and cropping
- OCR-based table extraction
- Reference-table detection
- Bibliography extraction
- DOI resolution using Crossref
- Structured CSV generation
- Interactive visualization UI
- Docker-based microservice architecture
- GPU-accelerated OCR processing

---

# Pipeline Workflow

```text
PDF Upload
    ↓
MinerU Table Detection
    ↓
Table Cropping
    ↓
PaddleOCR-VL Extraction
    ↓
Reference Table Detection
    ↓
GROBID Bibliography Extraction
    ↓
Reference Matching
    ↓
Crossref DOI Resolution
    ↓
Resolved CSV Generation
```

---

# Architecture

The system consists of multiple independent services communicating through REST APIs.

```text
React UI
    ↓
FastAPI Backend
    ├── MinerU Service
    ├── PaddleOCR-VL Service
    ├── GROBID
    └── Kreuzberg OCR Fallback
```

---

# Project Structure

```text
project-root/
│
├── backend/
├── ui_input/
├── mineru_service/
├── paddle_service/
├── grobid/
├── kreuzberg_service/
│
├── dataset/
├── evaluation/
│
├── docker-compose.yml
└── README.md
```

---

# Components

Each component contains its own dedicated README file with detailed implementation and setup documentation.

| Component | Description |
|---|---|
| `backend` | Main orchestration and API service |
| `ui_input` | React frontend for pipeline interaction |
| `mineru_service` | Table detection and cropping |
| `paddle_service` | OCR and table extraction |
| `grobid` | Bibliography extraction |
| `kreuzberg_service` | OCR fallback extraction |
| `dataset` | Evaluation dataset and benchmark results |
| `evaluation` | Evaluation plots and benchmark visualizations |

---

# Technologies

## Frontend

- React
- TypeScript
- Vite
- CSS Modules

## Backend

- FastAPI
- Python
- SQLAlchemy

## OCR & Extraction

- MinerU
- PaddleOCR-VL
- GROBID
- Kreuzberg OCR
- DeepSeek OCR 2
- Chandra OCR

## Infrastructure

- Docker
- Docker Compose
- GPU acceleration (CUDA)

---

# Evaluation

The project includes a complete evaluation framework for benchmarking:

- table extraction quality,
- OCR robustness,
- bibliography extraction,
- DOI matching,
- runtime efficiency.

Evaluation metrics include:

- RMS-based table similarity,
- precision,
- recall,
- F1-score,
- runtime analysis.

---

# Dataset

The repository includes references to the evaluation dataset used during the thesis work.

The dataset contains:

- scientific PDFs,
- ground truth tables,
- OCR outputs,
- bibliography extraction results,
- evaluation metrics,
- benchmark outputs.

The complete dataset exceeds 700 MB and is provided separately.

See:

```text
dataset/README.md
```

for detailed information.

---

# Evaluation Plots

Generated benchmark plots and visualizations are available in:

```text
evaluation/plots/
```

These plots include:

- OCR model comparisons,
- RMS extraction scores,
- runtime benchmarks,
- bibliography extraction metrics.

See:

```text
evaluation/README.md
```

for detailed information.

---

# UI

The project contains an interactive web UI for:

- uploading PDFs,
- visualizing cropped tables,
- viewing OCR extraction results,
- inspecting bibliography matches,
- downloading resolved CSV outputs.

---

# Intended Use

This project is intended for:

- scientific document processing,
- OCR benchmarking,
- research on table extraction,
- bibliography-aware NLP pipelines,
- structured scientific knowledge extraction.

---

# Notes

This repository contains experimental research software developed for scientific evaluation and benchmarking purposes.

The implementation prioritizes:

- reproducibility,
- modularity,
- transparency of intermediate outputs,
- evaluation support,
- extensibility for future research.