````md
# Evaluation Plots

This folder contains the generated evaluation plots and the scripts used to create them.

```text
plots/
├── reference_extraction/
├── table_extraction/
├── scripts/
└── README.md
````

Run all plot scripts from the `src/` directory:

```bash
cd src
```

Replace the dataset path with your local dataset location:

```bash
--dataset-root "D:\path\to\dataset"
```

---

## Reference Extraction Plots

### GROBID Mean F1

Creates a bar plot with mean precision, recall, and F1 score for GROBID.

```bash
python evaluation/plots/scripts/plot_grobid_mean_f1.py ^
  --dataset-root "D:\path\to\dataset" ^
  --out-dir "./evaluation/plots/reference_extraction"
```

### GROBID Numbered vs Other Patterns

Compares GROBID results for numbered references and other reference patterns.

```bash
python evaluation/plots/scripts/plot_grobid_numbered_vs_other.py ^
  --dataset-root "D:\path\to\dataset" ^
  --out-dir "./evaluation/plots/reference_extraction"
```

### Kreuzberg Mean F1

Creates a bar plot with mean precision, recall, and F1 score for the Kreuzberg + regex method.

```bash
python evaluation/plots/scripts/plot_kreuzberg_mean_f1.py ^
  --dataset-root "D:\path\to\dataset" ^
  --out-dir "./evaluation/plots/reference_extraction"
```

### Kreuzberg Numbered vs Other Patterns

Compares Kreuzberg + regex results for numbered references and other reference patterns.

```bash
python evaluation/plots/scripts/plot_kreuzberg_numbered_vs_other.py ^
  --dataset-root "D:\path\to\dataset" ^
  --out-dir "./evaluation/plots/reference_extraction"
```

### Reference Accuracy by Threshold

Creates line plots showing mean reference extraction accuracy across similarity thresholds.

```bash
python evaluation/plots/scripts/plot_reference_accuracy_runtime.py ^
  --dataset-root "D:\path\to\dataset" ^
  --out-dir "./evaluation/plots/reference_extraction"
```

---

## Table Extraction Plots

### Table Metrics by Category

Creates RMS precision, recall, and F1 plots grouped by table categories such as grid, size, cell density, and section.

```bash
python evaluation/plots/scripts/plot_table_metrics.py ^
  --out-dir "./evaluation/plots/table_extraction"
```

### Runtime Comparison

Creates a bar plot comparing mean runtime across OCR models.

```bash
python evaluation/plots/scripts/plot_runtime_comparison.py ^
  --dataset-root "D:\path\to\dataset" ^
  --out-dir "./evaluation/plots/table_extraction"
```

### Processed Tables per Model

Creates a plot showing how many tables each OCR model successfully processed or interpreted.

```bash
python evaluation/plots/scripts/plot_processed_tables.py ^
  --out-dir "./evaluation/plots/table_extraction"
```

### Ground Truth Table Distribution

Creates a plot showing the distribution of manually annotated ground-truth table categories.

```bash
python evaluation/plots/scripts/plot_ground_truth_distribution.py ^
  --out-dir "./evaluation/plots/table_extraction"
```

---

## Notes

* The plot scripts use Matplotlib.
* Generated plots are saved into either `reference_extraction/` or `table_extraction/`.
* Scripts using `--dataset-root` read JSON evaluation results from the dataset.
* Scripts without `--dataset-root` use manually defined aggregated values inside the script.

