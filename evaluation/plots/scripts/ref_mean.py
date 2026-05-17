import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

INPUT_JSON_NAME = "reference_eval_results.json"
OUTPUT_ACCURACY_PLOT_NAME = "mean_accuracy_all_models.png"
OUTPUT_TIME_PLOT_NAME = "mean_time_all_models.png"

EXTRA_MODEL_FILES = {
    "Kreuzberg OCR": "kreuzberg_ref_prediction_*.json",
}

MODEL_NAME_MAP = {
    "deepseek": "Deepseek OCR 2",
    "paddle": "Paddle VL 2",
    "runners": "Chandra 2",
    "kreuzberg": "Kreuzberg OCR",
}

model_colors = {
    "Deepseek OCR 2": "#4C72B0",
    "Paddle VL 2": "#55A868",
    "Chandra 2": "#8172B2",
    "Kreuzberg OCR": "#4C9F9F",
}


def normalize_model_name(name: str) -> str:
    name_lower = name.lower()

    for key, display_name in MODEL_NAME_MAP.items():
        if key in name_lower:
            return display_name

    return name


def get_time_value(stats: dict):
    """
    Tries common time field names.
    Adjust/add keys here if your JSON uses another name.
    """
    for key in ["time", "runtime", "elapsed_time", "duration", "seconds", "time_seconds"]:
        if key in stats and stats[key] is not None:
            return stats[key]
    return None


def collect_results(root: Path):
    accuracy_data = defaultdict(lambda: defaultdict(list))
    time_data = defaultdict(lambda: defaultdict(list))
    files_found = 0

    for json_file in root.rglob(INPUT_JSON_NAME):
        files_found += 1

        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Could not read {json_file}: {e}")
            continue

        models = data.get("models", {})

        for raw_model_name, model_data in models.items():
            model_name = normalize_model_name(raw_model_name)
            thresholds = model_data.get("thresholds", {})

            for threshold_str, stats in thresholds.items():
                threshold = float(threshold_str)

                accuracy = stats.get("accuracy")
                if accuracy is not None:
                    accuracy_data[model_name][threshold].append(float(accuracy))

                time_value = get_time_value(stats)
                if time_value is not None:
                    time_data[model_name][threshold].append(float(time_value))

        ref_folder = json_file.parent

        for model_name, pattern in EXTRA_MODEL_FILES.items():
            for extra_file in ref_folder.glob(pattern):
                try:
                    extra_data = json.loads(extra_file.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"Could not read {extra_file}: {e}")
                    continue

                thresholds = extra_data.get("thresholds", {})

                for threshold_str, stats in thresholds.items():
                    threshold = float(threshold_str)

                    accuracy = stats.get("accuracy")
                    if accuracy is not None:
                        accuracy_data[model_name][threshold].append(float(accuracy))

                    time_value = get_time_value(stats)
                    if time_value is not None:
                        time_data[model_name][threshold].append(float(time_value))

    return accuracy_data, time_data, files_found


def compute_means(aggregated):
    mean_results = {}

    for model_name, threshold_map in aggregated.items():
        mean_results[model_name] = {}

        for threshold, values in threshold_map.items():
            if values:
                mean_results[model_name][threshold] = sum(values) / len(values)

    return mean_results


def plot_results(mean_results, output_path: Path, ylabel: str, title: str, ylim=None):
    plt.figure(figsize=(10, 6))

    for model_name, threshold_map in sorted(mean_results.items()):
        thresholds = sorted(threshold_map.keys())
        means = [threshold_map[t] for t in thresholds]

        plt.plot(
            thresholds,
            means,
            marker="o",
            label=model_name,
            color=model_colors.get(model_name),
        )

    all_thresholds = sorted({t for m in mean_results.values() for t in m.keys()})

    plt.xlabel("Threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(all_thresholds)

    if ylim:
        plt.ylim(*ylim)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def main():
    root = Path(r"D:\Master\MasterArbeit\Issues\tabulus\data\dataset")

    if not root.exists() or not root.is_dir():
        print("Invalid folder path.")
        return

    accuracy_data, time_data, files_found = collect_results(root)

    if files_found == 0:
        print(f"No '{INPUT_JSON_NAME}' files found under: {root}")
        return

    mean_accuracy = compute_means(accuracy_data)
    mean_time = compute_means(time_data)

    if mean_accuracy:
        accuracy_output = root / OUTPUT_ACCURACY_PLOT_NAME
        plot_results(
            mean_accuracy,
            accuracy_output,
            ylabel="Mean Accuracy",
            title="Mean Accuracy by Model Across All Thresholds",
            ylim=(0, 1.05),
        )
        print(f"Accuracy plot saved to: {accuracy_output}")

    if mean_time:
        time_output = root / OUTPUT_TIME_PLOT_NAME
        plot_results(
            mean_time,
            time_output,
            ylabel="Mean Time",
            title="Mean Processing Time by Model Across All Thresholds",
        )
        print(f"Time plot saved to: {time_output}")
    else:
        print("No time values found. Check the time key inside your JSON files.")

    print(f"\nJSON files found: {files_found}")

    print("\nMean accuracies:")
    for model_name, threshold_map in sorted(mean_accuracy.items()):
        print(f"\n{model_name}")
        for threshold, mean_acc in sorted(threshold_map.items()):
            print(f"  threshold={threshold:.2f} -> mean_accuracy={mean_acc:.4f}")

    print("\nMean times:")
    for model_name, threshold_map in sorted(mean_time.items()):
        print(f"\n{model_name}")
        for threshold, mean_time_value in sorted(threshold_map.items()):
            print(f"  threshold={threshold:.2f} -> mean_time={mean_time_value:.4f}")


if __name__ == "__main__":
    main()