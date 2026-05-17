# Evaluation

This directory contains all evaluation plots and benchmark visualizations generated during the experimental evaluation of the scientific table extraction pipeline.

The evaluation focuses on:

- table extraction quality,
- OCR model comparison,
- bibliography extraction performance,
- reference matching accuracy,
- runtime efficiency.

The experiments were conducted on a manually curated dataset of scientific papers containing tables with references.

---

# Evaluation Metrics

The evaluation uses four main metrics:

- RMS-based table similarity,
- Normal Accuracy,
- Standard F1-score,
- Runtime.

Different metrics were required because the pipeline contains multiple extraction stages with different output formats and evaluation requirements.

---

# Runtime

Runtime is measured to compare the processing efficiency of the evaluated OCR models and extraction methods.

For each tool, the execution time is recorded from the start of the processing step until the final output file is generated.

```text
Runtime = t_end - t_start
```

This metric is especially important because some OCR models achieve higher extraction quality but require significantly more processing time.

The runtime evaluation is mainly used for:

- OCR model comparison,
- scalability analysis,
- processing efficiency benchmarking.

---

# RMS-Based Table Similarity

For table extraction evaluation, an RMS-based similarity metric from the DePlot library is used.

This metric compares:

- extracted table structure,
- extracted table content,
- ground truth tables.

The RMS-based score measures how closely the extracted table matches the expected table representation.

This evaluation is particularly suitable for:

- structured OCR benchmarking,
- table reconstruction quality,
- comparison of OCR extraction performance.

---

# Normal Accuracy

Normal accuracy is used for evaluating reference extraction from raw bibliography text.

For this evaluation:

- manually created ground truth reference JSON files are used,
- each expected reference string is searched in the extracted raw text,
- successfully detected references are counted as correct matches.

```text
Accuracy = Found References / Total References
```

This metric provides a simple measurement of how many expected references were successfully detected.

---

# Standard F1-Score

In addition to normal accuracy, the standard F1-score is used to evaluate the overall quality of extracted references.

The F1-score combines:

- precision,
- recall.

This is important because the evaluation should consider both:

- missing references,
- incorrectly extracted references.

## Precision

```text
Precision = TP / (TP + FP)
```

## Recall

```text
Recall = TP / (TP + FN)
```

## F1-Score

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Where:

- TP = correctly extracted references,
- FP = incorrectly extracted references,
- FN = references missing from the extraction output.

---

# Plot Structure

The evaluation plots are divided into two main categories.

---

# Reference Extraction

Location:

```text
plots/reference_extraction/
```

These plots evaluate:

- bibliography extraction quality,
- reference extraction accuracy,
- GROBID performance,
- regex-based fallback methods,
- DOI matching quality.

Included plots:

| Plot                                          | Description |
|-----------------------------------------------|---|
| `grobid_mean_f1_score_plot.png`               | Mean F1-score comparison for GROBID extraction |
| `grobid_numbered_vs_other_scores.png`         | Comparison between numbered and non-numbered reference structures |
| `kreuzberg_mean_accuracy_all_models.png`      | Average reference extraction accuracy across all evaluated models |
| `kreuzberg_mean_f1_score_plot.png`            | Overall F1-score comparison |
| `kreuzberg_numbered_vs_other_mean_scores.png` | Mean scores grouped by bibliography structure type |

---

# Table Extraction

Location:

```text
plots/table_extraction/
```

These plots evaluate:

- table OCR quality,
- table structure reconstruction,
- runtime performance,
- robustness against different table layouts.

Included plots:

| Plot | Description |
|---|---|
| `cell_density_model_metrics.png` | OCR performance grouped by table cell density |
| `column_hierarchy_model_metrics.png` | Performance comparison for tables with hierarchical columns |
| `grid_model_metrics.png` | Performance comparison for grid-based tables |
| `ground_truth_grouped_distribution.png` | Distribution of ground truth table categories |
| `mean_runtime_comparison.png` | Runtime comparison of evaluated OCR models |
| `overall_rms_model_metrics.png` | Overall RMS-based extraction performance |
| `plot_type_1_processed_tables.png` | Visualization of processed table types |
| `section_model_metrics.png` | Performance grouped by table section structure |
| `size_model_metrics.png` | OCR performance grouped by table size |

---

# Purpose of the Evaluation

The evaluation aims to analyze:

- extraction quality of OCR models,
- robustness against different table structures,
- bibliography extraction reliability,
- DOI enrichment quality,
- runtime efficiency,
- strengths and weaknesses of different extraction approaches.

The evaluation results are used to determine the most suitable OCR and bibliography extraction strategies for scientific table processing pipelines.

---
Add this section at the end of the evaluation README:

---

# Running the Evaluation Scripts

All evaluation scripts should be executed from the `src/` directory.

Example:

```bash
cd src
```

---

## Important

Most evaluation scripts require the dataset root path.

Before running the scripts, update the dataset path to the location where the evaluation dataset is stored on your machine.

Example:

```bash
--dataset-root "D:\path\to\dataset"
```

Linux example:

```bash
--dataset-root "/home/user/dataset"
```

---

# Table Extraction Evaluation

## Single CSV Evaluation

Compares one prediction CSV against one ground-truth CSV using the DePlot RMS-based metric.

```bash
python evaluation/evaluate_table_csv.py ^
  "D:\dataset\gold.csv" ^
  "D:\dataset\prediction.csv"
```

---

## Run All Table Evaluations

Runs evaluation for all prediction folders across the dataset.

Supported prediction folders:

* `deepseek2_prediction`
* `paddle_vl_prediction`
* `chandra_prediction`
* `Kreuzberg_prediction`

```bash
python evaluation/run_all_table_evaluations.py ^
  --dataset-root "D:\Master\MasterArbeit\Issues\tabulus\data\dataset"
```

---

# GROBID Reference Evaluation

## Fuzzy Reference F1 Evaluation

Evaluates GROBID bibliography extraction using fuzzy similarity matching.

```bash
python evaluation/evaluate_grobid_references_fuzzy.py ^
  --dataset-root "D:\Master\MasterArbeit\Issues\tabulus\data\dataset"
```

The script automatically searches for:

```text
gold_standard_references.json
grobid_prediction.json
```

and generates:

```text
grobid_fuzzy_f1_score_result.json
```

---

# Kreuzberg + Regex Evaluation

## Exact Reference F1 Evaluation

Evaluates extracted references using exact `(nr, ref)` matching.

```bash
python evaluation/evaluate_kreuzberg_references_exact.py ^
  --dataset-root "D:\Master\MasterArbeit\Issues\tabulus\data\dataset"
```

The script automatically searches for:

```text
gold_standard_references.json
extracted_kreuzberg_ref.json
```

and generates:

```text
kreuzberg_plus_regex_f1_score_result.json
```

---

# Raw Reference Text Similarity Evaluation

Evaluates whether references from the ground truth can be found inside raw OCR-extracted bibliography text using sliding-window similarity matching.

```bash
python evaluation/evaluate_reference_raw_text_similarity.py ^
  --dataset-root "D:\Master\MasterArbeit\Issues\tabulus\data\dataset"
```

Generated output:

```text
reference_eval_results.json
```

---

# Table Reference Coverage Evaluation

Evaluates whether extracted references cover the references that actually appear inside a table.

```bash
python evaluation/evaluate_table_reference_coverage.py ^
  --extracted "D:\...\extracted_refs.json" ^
  --gold "D:\...\gold_standard_references.json" ^
  --table-refs "D:\...\table_refs.json" ^
  --out "D:\...\kreuzberg_ref_eval.json"
```

---

# DePlot Library

The RMS-based table evaluation uses the DePlot evaluation library.

Location:

```text
evaluation/deplot/
```

The DePlot metric is used for:

* table structure similarity,
* table content similarity,
* OCR table reconstruction evaluation.

---

# Notes

* Generated evaluation files are written into the dataset folders.
* Large generated outputs should not be committed to the repository.
* Some scripts may require substantial runtime depending on dataset size and OCR outputs.
* The evaluation scripts were primarily developed and tested on Windows environments. 
 dataset used for evaluation contains manually curated scientific papers with annotated tables and bibliography references.