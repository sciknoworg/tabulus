import json
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

DATASET_PATH = Path(r"D:\Master\MasterArbeit\Issues\tabulus\data\dataset")

SCORE_FILENAME = "f1_score_result.json"
OUTPUT_JSON = "mean_f1_score_result.json"
OUTPUT_PLOT = "mean_f1_score_plot.png"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    precision_values = []
    recall_values = []
    f1_values = []
    processed_files = []

    for score_file in DATASET_PATH.rglob(SCORE_FILENAME):
        data = load_json(score_file)

        precision_values.append(data.get("precision", 0))
        recall_values.append(data.get("recall", 0))
        f1_values.append(data.get("f1_score", 0))

        processed_files.append(str(score_file))

    mean_precision = mean(precision_values) if precision_values else 0
    mean_recall = mean(recall_values) if recall_values else 0
    mean_f1 = mean(f1_values) if f1_values else 0

    result = {
        "description": "Mean F1 (GT vs prediction)",
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1_score": mean_f1,
        "evaluated_files": len(f1_values),
        "processed_files": processed_files
    }

    # Save JSON
    output_json_path = DATASET_PATH / OUTPUT_JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Saved JSON: {output_json_path}")

    # ---- CREATE PLOT ----
    labels = ["Precision", "Recall", "F1 Score"]
    values = [mean_precision, mean_recall, mean_f1]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)

    plt.ylim(0, 1)
    plt.title("Mean Evaluation Scores")
    plt.ylabel("Score")

    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.tight_layout()

    output_plot_path = DATASET_PATH / OUTPUT_PLOT
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_plot_path}")


if __name__ == "__main__":
    main()