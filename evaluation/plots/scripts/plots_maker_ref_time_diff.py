import json
from pathlib import Path
import matplotlib.pyplot as plt

CHANDRA_MEAN_RUNTIME = 7.872340

OUTPUT_PLOT_NAME = "mean_runtime_comparison.png"

model_colors = {
    "Deepseek OCR 2": "#4C72B0",
    "Paddle VL 2": "#55A868",
    "Chandra 2": "#8172B2",
    "Kreuzberg OCR": "#4C9F9F",
}


def collect_runtimes(root: Path):
    deepseek_runtimes = []
    paddle_runtimes = []
    kreuzberg_runtimes = []

    for json_file in root.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        runtime = data.get("runtime_seconds")

        if runtime is None:
            runtime = data.get("duration_seconds")

        if runtime is None:
            continue

        filename_lower = json_file.name.lower()
        path_lower = str(json_file).lower()

        if "deepseek" in filename_lower:
            deepseek_runtimes.append(float(runtime))
        elif "paddle" in filename_lower:
            paddle_runtimes.append(float(runtime))
        elif "kreuzberg" in filename_lower or "kreuzberg_prediction" in path_lower:
            kreuzberg_runtimes.append(float(runtime))

    return deepseek_runtimes, paddle_runtimes, kreuzberg_runtimes


def mean(values):
    return sum(values) / len(values) if values else None


def plot_runtime_comparison(deepseek_mean, paddle_mean, chandra_mean, kreuzberg_mean, output_path: Path):
    models = []
    runtimes = []

    if deepseek_mean is not None:
        models.append("Deepseek OCR 2")
        runtimes.append(deepseek_mean)

    if paddle_mean is not None:
        models.append("Paddle VL 2")
        runtimes.append(paddle_mean)

    models.append("Chandra 2")
    runtimes.append(chandra_mean)

    if kreuzberg_mean is not None:
        models.append("Kreuzberg OCR")
        runtimes.append(kreuzberg_mean)

    colors = [model_colors[m] for m in models]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, runtimes, color=colors)

    plt.xlabel("Model")
    plt.ylabel("Mean Runtime (seconds)")
    plt.title("Mean Runtime Comparison")

    for bar, value in zip(bars, runtimes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}s",
            ha="center",
            va="bottom"
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def main():
    root = Path(r"D:\Master\MasterArbeit\Issues\tabulus\data\dataset")

    if not root.exists() or not root.is_dir():
        print("Invalid folder path.")
        return

    deepseek_runtimes, paddle_runtimes, kreuzberg_runtimes = collect_runtimes(root)

    deepseek_mean = mean(deepseek_runtimes)
    paddle_mean = mean(paddle_runtimes)
    kreuzberg_mean = mean(kreuzberg_runtimes)
    chandra_mean = CHANDRA_MEAN_RUNTIME

    print("Runtime summary:")

    if deepseek_mean is not None:
        print(f"DeepSeek mean runtime: {deepseek_mean:.6f} s")
    else:
        print("DeepSeek runtime data not found.")

    if paddle_mean is not None:
        print(f"Paddle mean runtime: {paddle_mean:.6f} s")
    else:
        print("Paddle runtime data not found.")

    if kreuzberg_mean is not None:
        print(f"Kreuzberg mean runtime: {kreuzberg_mean:.6f} s")
        print(f"Kreuzberg - Chandra difference: {kreuzberg_mean - chandra_mean:.6f} s")
    else:
        print("Kreuzberg runtime data not found.")

    print(f"Chandra mean runtime: {chandra_mean:.6f} s")

    output_path = root / OUTPUT_PLOT_NAME
    plot_runtime_comparison(
        deepseek_mean,
        paddle_mean,
        chandra_mean,
        kreuzberg_mean,
        output_path
    )

    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()