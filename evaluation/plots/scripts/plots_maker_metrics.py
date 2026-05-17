from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_category_metrics_for_models(
    data: dict,
    category_name: str,
    output_dir: str | Path,
    filename: str | None = None,
    title: str | None = None,
):
    """
    Creates one plot for one category.

    Expected data format:
    {
        "Deepseek OCR 2": {
            "yes": {"precision": 0.91, "recall": 0.88, "f1": 0.89},
            "no":  {"precision": 0.95, "recall": 0.96, "f1": 0.95},
        },
        "Paddle VL 2": {
            "yes": {"precision": 0.93, "recall": 0.90, "f1": 0.91},
            "no":  {"precision": 0.97, "recall": 0.98, "f1": 0.97},
        },
        "Chandra 2": {
            "yes": {"precision": 0.89, "recall": 0.85, "f1": 0.87},
            "no":  {"precision": 0.94, "recall": 0.95, "f1": 0.94},
        },
    }
    """

    if not data:
        raise ValueError("data must not be empty.")

    output_path = ensure_output_dir(output_dir)

    if filename is None:
        safe_name = category_name.lower().replace(" ", "_")
        filename = f"{safe_name}_model_metrics.png"

    if title is None:
        title = f"{category_name}: RMS (cell-level): Precision / Recall / F1"

    models = list(data.keys())
    subclasses = list(next(iter(data.values())).keys())
    metrics = ["precision", "recall", "f1"]

    # neutral colors per model
    model_colors = {
        "Deepseek OCR 2": "#4C72B0",
        "Paddle VL 2": "#55A868",
        "Chandra 2": "#8172B2",
        "Kreuzberg OCR": "#4C9F9F",
    }

    fallback_colors = ["#4C72B0", "#55A868", "#8172B2", "#64B5CD", "#8C8C8C"]

    # x positions: one group per subclass
    x = np.arange(len(subclasses))

    # total bars in one subclass group = number of models * number of metrics
    n_models = len(models)
    n_metrics = len(metrics)
    total_bars_per_group = n_models * n_metrics
    bar_width = 0.8 / total_bars_per_group

    plt.figure(figsize=(12, 6))

    legend_added = set()

    for model_idx, model in enumerate(models):
        color = model_colors.get(model, fallback_colors[model_idx % len(fallback_colors)])

        for metric_idx, metric in enumerate(metrics):
            offset_index = model_idx * n_metrics + metric_idx
            offsets = x - 0.4 + (offset_index + 0.5) * bar_width

            values = [data[model][subclass][metric] for subclass in subclasses]

            # hatch patterns to distinguish precision / recall / f1
            hatch = {
                "precision": "",
                "recall": "//",
                "f1": "..",
            }[metric]

            label = f"{model} ({metric})"

            plt.bar(
                offsets,
                values,
                width=bar_width,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.4,
                label=label
            )

            for x_pos, value in zip(offsets, values):
                plt.text(
                    x_pos,
                    value + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

    plt.xticks(x, subclasses)
    plt.ylim(0, 1.1)
    plt.xlabel("Subclass")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()

    save_path = output_path / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {save_path}")


def plot_all_categories(all_category_data: dict, output_dir: str | Path):
    """
    Expected format:
    {
        "Grid": { ... model/subclass/metric data ... },
        "Column hierarchy": { ... },
        "Cell density": { ... },
        ...
    }
    """
    for category_name, category_data in all_category_data.items():
        plot_category_metrics_for_models(
            data=category_data,
            category_name=category_name,
            output_dir=output_dir
        )


if __name__ == "__main__":
    output_dir = r"D:\Master\MasterArbeit\Issues\tabulus\data\dataset\evaluation\plots"

    all_category_data = {
        "Grid": {
            "Kreuzberg OCR": {
                "yes": {
                    "precision": 0.2688,
                    "recall": 0.5095,
                    "f1": 0.3115
                },
                "no": {
                    "precision": 0.4107,
                    "recall": 0.6251,
                    "f1": 0.4482
                },
            },
            "Deepseek OCR 2": {
                "yes": {
                    "precision": 0.6118,
                    "recall": 0.6661,
                    "f1": 0.6208
                },
                "no": {
                    "precision": 0.6197,
                    "recall": 0.6281,
                    "f1": 0.6063
                },
            },
            "Paddle VL 2": {
                "yes": {
                    "precision": 0.6911,
                    "recall": 0.7895,
                    "f1": 0.7153
                },
                "no": {
                    "precision": 0.7142,
                    "recall": 0.7670,
                    "f1": 0.7261
                },
            },
            "Chandra 2": {
                "yes": {
                    "precision": 0.6813,
                    "recall": 0.6948,
                    "f1": 0.6784
                },
                "no": {
                    "precision": 0.7647,
                    "recall": 0.7592,
                    "f1": 0.7570
                },
            },
        },
        "Column hierarchy": {
            "Kreuzberg OCR": {
                "yes": {
                    "precision": 0.4257,
                    "recall": 0.6607,
                    "f1": 0.4755
                },
                "no": {
                    "precision": 0.3508,
                    "recall": 0.5751,
                    "f1": 0.3898
                },
            },
            "Deepseek OCR 2": {
                "yes": {
                    "precision": 0.5252,
                    "recall": 0.5143,
                    "f1": 0.5031
                },
                "no": {
                    "precision": 0.6215,
                    "recall": 0.6506,
                    "f1": 0.6182
                },
            },
            "Paddle VL 2": {
                "yes": {
                    "precision": 0.6586,
                    "recall": 0.7157,
                    "f1": 0.6776
                },
                "no": {
                    "precision": 0.7075,
                    "recall": 0.7792,
                    "f1": 0.7242
                },
            },
            "Chandra 2": {
                "yes": {
                    "precision": 0.7442,
                    "recall": 0.6678,
                    "f1": 0.7017
                },
                "no": {
                    "precision": 0.7304,
                    "recall": 0.7368,
                    "f1": 0.7266
                },
            },
        },
        "Cell density": {
            "Kreuzberg OCR": {
                "dense": {
                    "precision": 0.191,
                    "recall": 0.4571,
                    "f1": 0.2456
                },
                "simple": {
                    "precision": 0.5283,
                    "recall": 0.7093,
                    "f1": 0.5517
                },
            },

            "Deepseek OCR 2": {
                "dense": {
                    "precision": 0.5873,
                    "recall": 0.6284,
                    "f1": 0.5912
                },
                "simple": {
                    "precision": 0.6465,
                    "recall": 0.6589,
                    "f1": 0.6337
                },
            },
            "Paddle VL 2": {
                "dense": {
                    "precision": 0.6758,
                    "recall": 0.7504,
                    "f1": 0.6936
                },
                "simple": {
                    "precision": 0.7351,
                    "recall": 0.8023,
                    "f1": 0.7509
                },
            },
            "Chandra 2": {
                "dense": {
                    "precision": 0.6873,
                    "recall": 0.6912,
                    "f1": 0.6829
                },
                "simple": {
                    "precision": 0.7747,
                    "recall": 0.7751,
                    "f1": 0.7676
                },
            },
        },
        "Section": {
            "Kreuzberg OCR": {
                "yes": {
                    "precision": 0.4648,
                    "recall": .8522,
                    "f1": 0.5698
                },
                "no": {
                    "precision": 0.3471,
                    "recall": 0.5474,
                    "f1": 0.3822
                },
            },
            "Deepseek OCR 2": {
                "yes": {
                    "precision": 0.5322,
                    "recall": 0.5208,
                    "f1": 0.5126
                },
                "no": {
                    "precision": 0.6219,
                    "recall": 0.6513,
                    "f1": 0.6186
                },
            },
        "Paddle VL 2": {
            "yes": {
                "precision": 0.7016,
                "recall": 0.7164,
                "f1": 0.6972
            },
            "no": {
                "precision": 0.7052,
                "recall": 0.7796,
                "f1": 0.7233
            },
        },
        "Chandra 2": {
            "yes": {
                "precision": 0.5262,
                "recall": 0.5149,
                "f1": 0.5177
            },
            "no": {
                "precision": 0.7435,
                "recall": 0.7464,
                "f1": 0.7379
            },
        },
    },
        "Size": {
            "Kreuzberg OCR": {
                    "small": {
                        "precision": 0.3688,
                        "recall": 0.4784,
                        "f1": 0.3798
                    },
                    "medium": {
                        "precision": 0.406,
                        "recall": 0.6128,
                        "f1": 0.4529
                    },
                    "large": {
                        "precision": 0.2083,
                        "recall": 0.6696,
                        "f1": 0.2783
                    },
                },
            "Deepseek OCR 2": {
                "small": {
                    "precision": 0.6723,
                    "recall": 0.7100,
                    "f1": 0.6747
                },
                "medium": {
                    "precision": 0.5994,
                    "recall": 0.6191,
                    "f1": 0.5931
                },
                "large": {
                    "precision": 0.5563,
                    "recall": 0.5818,
                    "f1": 0.5443
                },
            },
            "Paddle VL 2": {
                "small": {
                    "precision": 0.7253,
                    "recall": 0.7733,
                    "f1": 0.7347
                },
                "medium": {
                    "precision": 0.6969,
                    "recall": 0.7724,
                    "f1": 0.7168
                },
                "large": {
                    "precision": 0.6885,
                    "recall": 0.7902,
                    "f1": 0.7108
                },
            },
            "Chandra 2": {
                "small": {
                    "precision": 0.7786,
                    "recall": 0.7833,
                    "f1": 0.7697
                },
                "medium": {
                    "precision": 0.7055,
                    "recall": 0.7041,
                    "f1": 0.7015
                },
                "large": {
                    "precision": 0.7132,
                    "recall": 0.7195,
                    "f1": 0.7084
                },
            },
        },
        "": {
            "Kreuzberg OCR": {
                "": {
                    "precision": 0.33975,
                    "recall": 0.5673,
                    "f1": 0.37985
                }
            },
            "Deepseek OCR 2": {
                "": {
                    "precision": 0.61575,
                    "recall": 0.6471,
                    "f1": 0.61355
                }
            },
            "Paddle VL 2": {
                "": {
                    "precision": 0.70265,
                    "recall": 0.77825,
                    "f1": 0.7207
                }
            },
            "Chandra 2": {
                "": {
                    "precision": 0.723,
                    "recall": 0.727,
                    "f1": 0.7177
                }
            }
        }


    }

    plot_all_categories(all_category_data, output_dir)