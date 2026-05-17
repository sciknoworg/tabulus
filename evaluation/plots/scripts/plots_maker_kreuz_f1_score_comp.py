import csv
import json
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

DATASET_PATH = Path(r"D:\Master\MasterArbeit\Issues\tabulus\data\dataset")

PATTERN_CSV = "kreuzberg_pattern_summary.csv"
SCORE_FILENAME = "f1_score_result.json"

OUTPUT_JSON = "numbered_vs_other_mean_scores.json"
OUTPUT_PLOT = "numbered_vs_other_mean_scores.png"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pattern_csv(csv_path):
    pattern_by_folder = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            file_path = Path(row["file_path"])
            ref_folder = file_path.parent

            pattern_by_folder[str(ref_folder)] = row["pattern_used"].strip().lower()

    return pattern_by_folder


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

        group_name = "Numbered" if pattern_used == "numbered" else "Other patterns"

        groups[group_name]["precision"].append(data.get("precision", 0))
        groups[group_name]["recall"].append(data.get("recall", 0))
        groups[group_name]["f1_score"].append(data.get("f1_score", 0))

    result = {}

    for group_name, values in groups.items():
        result[group_name] = {
            "mean_precision": mean(values["precision"]) if values["precision"] else 0,
            "mean_recall": mean(values["recall"]) if values["recall"] else 0,
            "mean_f1_score": mean(values["f1_score"]) if values["f1_score"] else 0,
            "evaluated_files": len(values["f1_score"])
        }

    output_json_path = DATASET_PATH / OUTPUT_JSON

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Saved JSON: {output_json_path}")

    # ---------- PLOT ----------
    metrics = ["Precision", "Recall", "F1 Score"]

    numbered_values = [
        result["Numbered"]["mean_precision"],
        result["Numbered"]["mean_recall"],
        result["Numbered"]["mean_f1_score"],
    ]

    other_values = [
        result["Other patterns"]["mean_precision"],
        result["Other patterns"]["mean_recall"],
        result["Other patterns"]["mean_f1_score"],
    ]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(7, 4))

    bars1 = plt.bar(
        [i - width / 2 for i in x],
        numbered_values,
        width,
        label="Numbered",
        color="#4C72B0"  # blue
    )

    bars2 = plt.bar(
        [i + width / 2 for i in x],
        other_values,
        width,
        label="Other patterns",
        color="#6C757D"  # gray
    )
    plt.xticks(list(x), metrics)
    plt.ylim(0, 1)
    plt.ylabel("Mean Score")
    plt.title("Numbered Pattern vs Other Patterns")
    plt.legend()

    for bars in [bars1, bars2]:
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
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()