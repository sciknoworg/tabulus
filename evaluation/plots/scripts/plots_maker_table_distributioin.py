from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_grouped_category_distribution(
    data: dict,
    output_dir: str | Path,
    filename: str = "ground_truth_grouped_distribution.png",
    title: str = "Ground Truth Table Distribution by Category",
):
    if not data:
        raise ValueError("data must not be empty.")

    output_path = ensure_output_dir(output_dir)

    category_colors = [
        "#4C72B0",  # muted blue
        "#55A868",  # muted green
        "#8172B2",  # muted purple
        "#64B5CD",  # soft cyan-blue
        "#8C8C8C",  # gray
        "#4D4D4D",  # dark gray
    ]

    x_positions = []
    x_labels = []
    bar_values = []
    bar_colors = []

    legend_handles = []
    current_x = 0
    gap_between_groups = 1.2

    categories = list(data.keys())

    for i, category in enumerate(categories):
        subclasses = data[category]
        color = category_colors[i % len(category_colors)]

        legend_handles.append(Patch(facecolor=color, label=category))

        for subclass, count in subclasses.items():
            x_positions.append(current_x)
            x_labels.append(subclass)
            bar_values.append(count)
            bar_colors.append(color)
            current_x += 1

        current_x += gap_between_groups

    plt.figure(figsize=(12, 6))
    plt.bar(x_positions, bar_values, color=bar_colors)

    y_offset = max(max(bar_values) * 0.02, 1) if bar_values else 1

    for x, value in zip(x_positions, bar_values):
        plt.text(
            x,
            value + y_offset,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(x_positions, x_labels, rotation=35, ha="right")
    plt.xlabel("Subclasses")
    plt.ylabel("Number of Tables")
    plt.title(title)
    plt.legend(handles=legend_handles, title="Category")

    max_y = max(bar_values) if bar_values else 0
    plt.ylim(0, max_y * 1.15 if max_y > 0 else 1)

    plt.tight_layout()

    save_path = output_path / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    output_dir = r"D:\Master\MasterArbeit\Issues\tabulus\data\dataset\evaluation\plots"

    data = {
        "Grid": {"yes": 328, "no": 212},
        "Column hierarchy": {"yes": 28, "no": 512},
        "Cell density": {"dense": 278, "simple": 262},
        "Section": {"yes": 34, "no": 506},
        "Size": {"small": 179, "medium": 255, "large": 106},
    }

    plot_grouped_category_distribution(
        data=data,
        output_dir=output_dir,
        filename="ground_truth_grouped_distribution.png",
        title="Ground Truth Table Distribution by Category",
    )