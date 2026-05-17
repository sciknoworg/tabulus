import csv
import json
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

DATASET_PATH = Path(r"D:\Master\MasterArbeit\Issues\tabulus\data\dataset")

PATTERN_CSV = "kreuzberg_pattern_summary.csv"
SCORE_FILENAME = "grobid_fuzzy_f1_score_result.json"

OUTPUT_JSON = "grobid_numbered_vs_other_scores.json"
OUTPUT_PLOT = "grobid_numbered_vs_other_scores.png"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pattern_csv(csv_path):
    pattern_by_folder = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            file_path = Path(row["file_path"])
            ref_folder = str(file_path.parent)

            pattern_by_folder[ref_folder] = row["pattern_used"].strip().lower()

    return pattern_by_folder


def calculate_group_means(values):
    return {
        "precision": mean(values["precision"]) if values["precision"] else 0,
        "recall": mean(values["recall"]) if values["recall"] else 0,
        "f1_score": mean(values["f1_score"]) if values["f1_score"] else 0,
    }


def main():
    csv_path = DATASET_PATH / PATTERN_CSV
    pattern_by_folder = load_pattern_csv(csv_path)

    groups = {
        "Numbered": {
            "precision": [],
            "recall": [],
            "f1_score": []
        },
        "Other patterns": {
            "precision": [],
            "recall": [],
            "f1_score": []
        }
    }

    for score_file in DATASET_PATH.rglob(SCORE_FILENAME):
        ref_folder = str(score_file.parent)

        if ref_folder not in pattern_by_folder:
            continue

        pattern_used = pattern_by_folder[ref_folder]

        data = load_json(score_file)

        group_name = (
            "Numbered"
            if pattern_used == "numbered"
            else "Other patterns"
        )

        groups[group_name]["precision"].append(data.get("precision", 0))
        groups[group_name]["recall"].append(data.get("recall", 0))
        groups[group_name]["f1_score"].append(data.get("f1_score", 0))

    numbered = calculate_group_means(groups["Numbered"])
    other = calculate_group_means(groups["Other patterns"])

    diff = {
        "precision_diff": numbered["precision"] - other["precision"],
        "recall_diff": numbered["recall"] - other["recall"],
        "f1_diff": numbered["f1_score"] - other["f1_score"],
    }

    result = {
        "Numbered": numbered,
        "Other patterns": other,
        "Difference": diff
    }

    output_json_path = DATASET_PATH / OUTPUT_JSON

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Saved JSON: {output_json_path}")

    # ---------------- PLOT ----------------

    metrics = ["Precision", "Recall", "F1 Score"]

    numbered_values = [
        numbered["precision"],
        numbered["recall"],
        numbered["f1_score"]
    ]

    other_values = [
        other["precision"],
        other["recall"],
        other["f1_score"]
    ]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))

    bars1 = plt.bar(
        [i - width / 2 for i in x],
        numbered_values,
        width,
        label="Numbered",
        color="#4C72B0"
    )

    bars2 = plt.bar(
        [i + width / 2 for i in x],
        other_values,
        width,
        label="Other patterns",
        color="#6C757D"
    )

    plt.xticks(list(x), metrics)
    plt.ylim(0, 1)
    plt.ylabel("Mean Score")
    plt.title("GROBID: Numbered vs Other Patterns")
    plt.legend()

    # values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.2f}",
                ha="center",
                va="bottom"
            )

    # difference text
    diff_text = (
        f"Δ Precision: {diff['precision_diff']:.2f}\n"
        f"Δ Recall: {diff['recall_diff']:.2f}\n"
        f"Δ F1: {diff['f1_diff']:.2f}"
    )

    plt.figtext(
        0.72,
        0.18,
        diff_text,
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray")
    )

    plt.tight_layout()

    output_plot_path = DATASET_PATH / OUTPUT_PLOT

    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_plot_path}")


if __name__ == "__main__":
    main()