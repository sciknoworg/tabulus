from pathlib import Path
import matplotlib.pyplot as plt


def ensure_output_dir(output_dir: str | Path) -> Path:
    """
    Create output directory if it does not exist and return it as absolute Path.
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Using output dir: {output_path}")
    return output_path


def plot_processed_tables(
    model_data: dict,
    total_tables: int,
    output_dir: str | Path,
    filename: str = "processed_tables.png",
    title: str = "Processed / Interpreted Tables per Model",
):
    if not model_data:
        raise ValueError("model_data must not be empty.")

    if total_tables < 0:
        raise ValueError("total_tables must be >= 0.")

    output_path = ensure_output_dir(output_dir)

    model_names = list(model_data.keys())
    processed_values = list(model_data.values())

    for value in processed_values:
        if value < 0:
            raise ValueError("Processed table counts must be >= 0.")

    # ✅ neutral colors (no red/orange/yellow)
    colors = [
        "#4C72B0",  # muted blue
        "#55A868",  # muted green
        "#8172B2",  # muted purple
        "#64B5CD",  # soft cyan-blue (fallback if more models)
        "#8C8C8C",  # gray
    ]

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        model_names,
        processed_values,
        color=colors[:len(model_names)]
    )

    # horizontal reference line
    plt.axhline(
        y=total_tables,
        linestyle="--",
        color="#2F2F2F",
        label=f"All tables = {total_tables}"
    )

    max_y = max([total_tables] + processed_values)

    # labels INSIDE bars
    for bar, value in zip(bars, processed_values):
        percentage = (value / total_tables * 100) if total_tables > 0 else 0

        # dynamic placement (safe for small bars)
        y_pos = value - max_y * 0.05

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{value}\n({percentage:.1f}%)",
            ha="center",
            va="top",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    plt.ylim(0, max_y * 1.15)

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Number of Tables")
    plt.legend()
    plt.tight_layout()

    save_path = output_path / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {save_path}")
    print(f"Absolute file location: {save_path.resolve()}")


def plot_type_1(output_dir: str | Path):
    """
    Your actual data
    """
    total_tables = 540

    model_data = {
        "Deepseek OCR 2": 497,
        "Paddle VL 2": 524,
        "Chandra 2": 509,
        "Kreuzberg": 530
    }

    plot_processed_tables(
        model_data=model_data,
        total_tables=total_tables,
        output_dir=output_dir,
        filename="plot_type_1_processed_tables.png",
        title="Processed / Interpreted Tables per Model"
    )


if __name__ == "__main__":
    output_dir = r"D:\Master\MasterArbeit\Issues\tabulus\data\dataset\evaluation\plots"

    plot_type_1(output_dir)