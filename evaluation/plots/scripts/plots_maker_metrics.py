from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import numpy as np


plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "figure.facecolor": "#f4f6f8",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#d0d0d0",
    "axes.linewidth": 1.2,
    "axes.titleweight": "bold",
    "axes.titlesize": 17,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "grid.color": "#d9d9d9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.facecolor": "#f4f6f8",
})


category_descriptions = {
    "Grid":
        "Performance comparison of OCR models on tables with and without visible grid structures.",

    "Column hierarchy":
        "Performance comparison of OCR models on tables containing column hierarchies and multi-level column headers.",

    "Row hierarchy":
        "Performance comparison of OCR models on tables containing row hierarchies and nested row structures.",

    "Cell density":
        "Performance comparison of OCR models on tables with varying cell densities, ranging from simple to densely populated layouts.",

    "Section":
        "Performance comparison of OCR models on tables with and without section-based organization.",

    "Size":
        "Performance comparison of OCR models across tables of different sizes, including small, medium, and large layouts.",

    "":
        "Overall RMS cell-level precision, recall, and F1-score comparison across all evaluated OCR models."
}

model_colors = {
    "Kreuzberg OCR": "#4C9F9F",
    "Deepseek OCR 2": "#4C72B0",
    "Paddle VL 2": "#55A868",
    "Chandra 2": "#8172B2",
    "NuExtract3": "#B8860B",  # neutral gray
}


metric_alpha = {
    "precision": 0.45,
    "recall": 0.70,
    "f1": 1.00,
}


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def safe_filename(category_name: str) -> str:
    if category_name.strip() == "":
        return "overall_model_metrics.png"

    return (
        category_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        + "_model_metrics.png"
    )


def add_custom_legends(ax):
    handles = [
        Patch(facecolor=color, edgecolor="black", label=model)
        for model, color in model_colors.items()
    ]

    handles += [
        Patch(facecolor="none", edgecolor="none", label="   Metrics:"),
        Patch(facecolor="gray", edgecolor="black", alpha=0.45, label="Precision"),
        Patch(facecolor="gray", edgecolor="black", alpha=0.70, label="Recall"),
        Patch(facecolor="gray", edgecolor="black", alpha=1.00, label="F1 Score"),
    ]

    legend = ax.legend(
        handles=handles,
        title="OCR Models:",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(handles),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.4,
        columnspacing=1.0,
        handlelength=1.0,
        handletextpad=0.4,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cccccc")


def plot_category_metrics_for_models(
    data: dict,
    category_name: str,
    output_dir: str | Path,
    filename: str | None = None,
    title: str | None = None,
):
    if not data:
        raise ValueError("data must not be empty.")

    output_path = ensure_output_dir(output_dir)

    if filename is None:
        filename = safe_filename(category_name)

    display_category = category_name if category_name.strip() else "Overall"

    if title is None:
        title = f"{display_category}: RMS Cell-level Precision, Recall, and F1 Score"

    models = list(data.keys())
    subclasses = list(next(iter(data.values())).keys())
    metrics = ["precision", "recall", "f1"]

    x = np.arange(len(subclasses)) * 0.35

    n_models = len(models)
    n_metrics = len(metrics)
    total_bars_per_group = n_models * n_metrics

    bar_width =  0.020

    fig, ax = plt.subplots(figsize=(10, 5.5))

    description = category_descriptions.get(category_name, "Dummy description.")

    fig.text(
        0.5,
        0.93,
        description,
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
        style="italic",
        wrap=True,
        linespacing=1.4,
        bbox=dict(
            facecolor="#f8f9fa",
            edgecolor="#cfcfcf",
            linewidth=1.1,
            boxstyle="round,pad=0.55"
        )
    )

    for model_idx, model in enumerate(models):
        color = model_colors.get(model, "#8C8C8C")

        for metric_idx, metric in enumerate(metrics):
            offset_index = model_idx * n_metrics + metric_idx

            offsets = (
                x
                - (bar_width * total_bars_per_group) / 2
                + (offset_index + 0.5) * bar_width
            )

            values = [
                data[model][subclass][metric]
                for subclass in subclasses
            ]

            bars = ax.bar(
                offsets,
                values,
                width=bar_width,
                color=color,
                alpha=metric_alpha[metric],
                edgecolor="black",
                linewidth=0.55,
                zorder=3,
            )

            for bar in bars:
                bar.set_path_effects([
                    pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.22),
                    pe.Normal()
                ])

            for x_pos, value in zip(offsets, values):
                y_offset = 0.015 if value < 0.9 else 0.005

                ax.text(
                    x_pos,
                    value + y_offset,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#111111",
                    rotation=90,
                    zorder=5,
                    clip_on=False,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="#d0d0d0",
                        linewidth=0.5,
                        alpha=0.85,
                        boxstyle="round,pad=0.12"
                    )
                )

    ax.set_xticks(x)
    ax.set_xticklabels(subclasses)

    ax.set_ylim(0, 1.1)

    ax.set_xlabel("Subclass")
    ax.set_ylabel("Score")


    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.grid(axis="x", alpha=0.18, zorder=0)

    add_custom_legends(ax)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])

    save_path = output_path / filename

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved plot: {save_path}")


def plot_all_categories(all_category_data: dict, output_dir: str | Path):
    for category_name, category_data in all_category_data.items():
        plot_category_metrics_for_models(
            data=category_data,
            category_name=category_name,
            output_dir=output_dir,
        )
if __name__ == "__main__":
    output_dir = r"D:\Master\MasterArbeit\Issues\tabulus\data\dataset\evaluation\plots"

    all_category_data = {
        "Grid": {
            "Kreuzberg OCR": {
                "yes": {"precision": 0.0094, "recall": 0.0292, "f1": 0.0123},
                "no": {"precision": 0.0211, "recall": 0.0432, "f1": 0.0263},
            },
            "Deepseek OCR 2": {
                "yes": {"precision": 0.6645, "recall": 0.6953, "f1": 0.6608},
                "no": {"precision": 0.6343, "recall": 0.6248, "f1": 0.6092},
            },
            "Paddle VL 2": {
                "yes": {"precision": 0.7576, "recall": 0.8471, "f1": 0.7803},
                "no": {"precision": 0.7066, "recall": 0.7626, "f1": 0.7232},
            },
            "Chandra 2": {
                "yes": {"precision": 0.7294, "recall": 0.7329, "f1": 0.7260},
                "no": {"precision": 0.7582, "recall": 0.7453, "f1": 0.7452},
            },
            "NuExtract3": {
                "yes": {"precision": 0.8203, "recall": 0.8267, "f1": 0.8184},
                "no": {"precision": 0.8338, "recall": 0.8232, "f1": 0.8255},
            },
        },

        "Column hierarchy": {
            "Kreuzberg OCR": {
                "yes": {"precision": 0.0284, "recall": 0.0341, "f1": 0.0291},
                "no": {"precision": 0.0159, "recall": 0.0379, "f1": 0.0203},
            },
            "Deepseek OCR 2": {
                "yes": {"precision": 0.4069, "recall": 0.3618, "f1": 0.3374},
                "no": {"precision": 0.6594, "recall": 0.6690, "f1": 0.6458},
            },
            "Paddle VL 2": {
                "yes": {"precision": 0.3637, "recall": 0.4737, "f1": 0.3916},
                "no": {"precision": 0.7462, "recall": 0.8133, "f1": 0.7647},
            },
            "Chandra 2": {
                "yes": {"precision": 0.1356, "recall": 0.1889, "f1": 0.1526},
                "no": {"precision": 0.7789, "recall": 0.7694, "f1": 0.7683},
            },
            "NuExtract3": {
                "yes": {"precision": 0.6107, "recall": 0.6661, "f1": 0.6258},
                "no": {"precision": 0.8400, "recall": 0.8330, "f1": 0.8331},
            },
        },

        "Row hierarchy": {
            "Kreuzberg OCR": {
                "yes": {"precision": 0.0046, "recall": 0.0245, "f1": 0.0071},
                "no": {"precision": 0.0195, "recall": 0.0410, "f1": 0.0241},
            },
            "Deepseek OCR 2": {
                "yes": {"precision": 0.3337, "recall": 0.4374, "f1": 0.3596},
                "no": {"precision": 0.7231, "recall": 0.7062, "f1": 0.6963},
            },
            "Paddle VL 2": {
                "yes": {"precision": 0.3613, "recall": 0.5978, "f1": 0.4220},
                "no": {"precision": 0.8185, "recall": 0.8458, "f1": 0.8270},
            },
            "Chandra 2": {
                "yes": {"precision": 0.4522, "recall": 0.4451, "f1": 0.4429},
                "no": {"precision": 0.8213, "recall": 0.8152, "f1": 0.8122},
            },
            "NuExtract3": {
                "yes": {"precision": 0.5890, "recall": 0.5696, "f1": 0.5743},
                "no": {"precision": 0.8882, "recall": 0.8882, "f1": 0.8846},
            },
        },

        "Cell density": {
            "Kreuzberg OCR": {
                "dense": {"precision": 0.0102, "recall": 0.0273, "f1": 0.0131},
                "simple": {"precision": 0.0232, "recall": 0.0487, "f1": 0.0288},
            },
            "Deepseek OCR 2": {
                "dense": {"precision": 0.6405, "recall": 0.6587, "f1": 0.6303},
                "simple": {"precision": 0.6529, "recall": 0.6474, "f1": 0.6297},
            },
            "Paddle VL 2": {
                "dense": {"precision": 0.7304, "recall": 0.8101, "f1": 0.7527},
                "simple": {"precision": 0.7231, "recall": 0.7812, "f1": 0.7385},
            },
            "Chandra 2": {
                "dense": {"precision": 0.7518, "recall": 0.7424, "f1": 0.7423},
                "simple": {"precision": 0.7413, "recall": 0.7381, "f1": 0.7324},
            },
            "NuExtract3": {
                "dense": {"precision": 0.8290, "recall": 0.8249, "f1": 0.8225},
                "simple": {"precision": 0.8279, "recall": 0.8244, "f1": 0.8229},
            },
        },

        "Section": {
            "Kreuzberg OCR": {
                "yes": {"precision": 0.0247, "recall": 0.0546, "f1": 0.0317},
                "no": {"precision": 0.0159, "recall": 0.0366, "f1": 0.0200},
            },
            "Deepseek OCR 2": {
                "yes": {"precision": 0.6649, "recall": 0.6170, "f1": 0.6128},
                "no": {"precision": 0.6451, "recall": 0.6559, "f1": 0.6312},
            },
            "Paddle VL 2": {
                "yes": {"precision": 0.7855, "recall": 0.8419, "f1": 0.8060},
                "no": {"precision": 0.7229, "recall": 0.7930, "f1": 0.7418},
            },
            "Chandra 2": {
                "yes": {"precision": 0.6693, "recall": 0.6526, "f1": 0.6584},
                "no": {"precision": 0.7519, "recall": 0.7463, "f1": 0.7428},
            },
            "NuExtract3": {
                "yes": {"precision": 0.8678, "recall": 0.8476, "f1": 0.8560},
                "no": {"precision": 0.8258, "recall": 0.8231, "f1": 0.8204},
            },
        },

        "Size": {
            "Kreuzberg OCR": {
                "small": {"precision": 0.0113, "recall": 0.0262, "f1": 0.0152},
                "medium": {"precision": 0.0215, "recall": 0.0440, "f1": 0.0267},
                "large": {"precision": 0.0132, "recall": 0.0419, "f1": 0.0158},
            },
            "Deepseek OCR 2": {
                "small": {"precision": 0.7195, "recall": 0.7308, "f1": 0.7139},
                "medium": {"precision": 0.6404, "recall": 0.6401, "f1": 0.6167},
                "large": {"precision": 0.5350, "recall": 0.5510, "f1": 0.5171},
            },
            "Paddle VL 2": {
                "small": {"precision": 0.7598, "recall": 0.7992, "f1": 0.7724},
                "medium": {"precision": 0.7168, "recall": 0.7909, "f1": 0.7375},
                "large": {"precision": 0.6946, "recall": 0.8039, "f1": 0.7204},
            },
            "Chandra 2": {
                "small": {"precision": 0.7879, "recall": 0.7703, "f1": 0.7680},
                "medium": {"precision": 0.7320, "recall": 0.7256, "f1": 0.7266},
                "large": {"precision": 0.7149, "recall": 0.7269, "f1": 0.7139},
            },
            "NuExtract3": {
                "small": {"precision": 0.8527, "recall": 0.8532, "f1": 0.8511},
                "medium": {"precision": 0.8227, "recall": 0.8184, "f1": 0.8185},
                "large": {"precision": 0.8011, "recall": 0.7910, "f1": 0.7847},
            },
        },

        "": {
            "Kreuzberg OCR": {
                "": {"precision": 0.0165, "recall": 0.0377, "f1": 0.0207}
            },
            "Deepseek OCR 2": {
                "": {"precision": 0.6465, "recall": 0.6533, "f1": 0.6300}
            },
            "Paddle VL 2": {
                "": {"precision": 0.7269, "recall": 0.7961, "f1": 0.7458}
            },
            "Chandra 2": {
                "": {"precision": 0.7466, "recall": 0.7403, "f1": 0.7375}
            },
            "NuExtract3": {
                "": {"precision": 0.8284, "recall": 0.8246, "f1": 0.8227}
            },
        },
    }



    plot_all_categories(all_category_data, output_dir)