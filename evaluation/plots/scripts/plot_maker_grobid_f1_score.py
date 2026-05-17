import json
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

DATASET_PATH = Path(r"D:\Master\MasterArbeit\Issues\tabulus\data\dataset")

SCORE_FILENAME = "grobid_fuzzy_f1_score_result.json"

OUTPUT_JSON = "grobid_mean_f1_score_result.json"
OUTPUT_PLOT = "grobid_mean_f1_score_plot.png"


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
        "description": "Mean GROBID evaluation",
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1_score": mean_f1,
        "evaluated_files": len(f1_values),
        "processed_files": processed_files
    }

    output_json_path = DATASET_PATH / OUTPUT_JSON

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Saved JSON: {output_json_path}")

    labels = ["Precision", "Recall", "F1 Score"]
    values = [mean_precision, mean_recall, mean_f1]

    plt.figure(figsize=(6, 4))

    bars = plt.bar(
        labels,
        values,
        color=["#4C72B0", "#6C757D", "#2F4B7C"]
    )

    plt.ylim(0, 1)
    plt.title("Mean GROBID Evaluation Scores")
    plt.ylabel("Mean Score")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output_plot_path = DATASET_PATH / OUTPUT_PLOT
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_plot_path}")


if __name__ == "__main__":
    main()