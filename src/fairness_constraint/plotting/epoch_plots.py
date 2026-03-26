"""Generate epoch-wise metric plots (training curves for different parameter values).

Each plot shows how a metric evolves over epochs, with one line per
unique value of the "other" parameter (lambda or epsilon/delta).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from ..config import get_args
from ..parse import parse_results
from .config import (
    MODEL_CONFIG, FONT_SIZES, FIGURE_SIZE, GRID_ALPHA,
    apply_plot_style, format_value, generate_epoch_colors,
)

args = get_args()

# ---------------------------------------------------------------------------
# Metric definitions for epoch plots
# ---------------------------------------------------------------------------
METRICS = [
    dict(
        column="fairness_column",
        title="{dataset}",
        y_label=r"$\bar{r}_{dp}(w)$",
        filename="demographic_parity_measure_{fixed_param}_{fixed_value}",
        y_limits=None,
    ),
    dict(
        column="acc_overall",
        title="{dataset}",
        y_label="training accuracy (%)",
        filename="accuracy_{fixed_param}_{fixed_value}",
        y_limits=[0, 100],
    ),
]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
def _get_scenarios(model_type):
    """Return the (fixed_param, other_param, fixed_value) scenarios."""
    if model_type != 4:
        return [
            {"fixed_param": "epsilon", "other_param": "lmbd", "fixed_value": 0},
            {"fixed_param": "lmbd", "other_param": "epsilon", "fixed_value": float("inf")},
        ]
    return [
        {"fixed_param": "epsilon", "other_param": "lmbd", "fixed_value": 1.0e18},
        {"fixed_param": "lmbd", "other_param": "epsilon", "fixed_value": float("inf")},
    ]


def _param_symbol(other_param, model_type):
    """Return the display symbol for the varying parameter."""
    if other_param == "lmbd":
        return "\u03bb"
    return "\u03b5" if model_type == 4 else "\u03b4"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _create_epoch_plot(df_filtered, fixed_param, fixed_value, other_param,
                       model_config, metric_cfg, output_dir, model_type, dataset):
    """Render and save a single epoch-wise metric plot."""
    if df_filtered.empty:
        print(f"No data for {fixed_param}={fixed_value}. Skipping.")
        return

    # Resolve actual column name
    col = model_config["fairness_column"] if metric_cfg["column"] == "fairness_column" else metric_cfg["column"]
    if col not in df_filtered.columns:
        print(f"Column {col} not found. Skipping.")
        return

    sym = _param_symbol(other_param, model_type)
    colors = generate_epoch_colors(20)

    apply_plot_style()
    plt.figure(figsize=FIGURE_SIZE)

    for i, val in enumerate(sorted(df_filtered[other_param].unique())):
        subset = df_filtered[df_filtered[other_param] == val]
        if subset.empty:
            continue
        plt.plot(
            subset["epoch"],
            subset[col],
            label=f"{sym} = {format_value(val)}",
            linestyle=model_config["line_styles"][i % len(model_config["line_styles"])],
            color=colors[i % len(colors)],
            linewidth=2.5,
        )

    plt.title(metric_cfg["title"].format(dataset=dataset))
    plt.xlabel("epoch")
    plt.ylabel(metric_cfg["y_label"])
    if metric_cfg["y_limits"]:
        plt.ylim(*metric_cfg["y_limits"])
    plt.legend(loc="lower right", framealpha=0.5)
    plt.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()

    fixed_str = "inf" if fixed_value == float("inf") else format_value(fixed_value)
    fname = metric_cfg["filename"].format(fixed_param=fixed_param, fixed_value=fixed_str)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, fname)}.png")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def plot_model_type(model_type, args, out_dir):
    """Generate epoch-wise plots for all scenarios of a model type."""
    cfg = MODEL_CONFIG.get(model_type)
    if not cfg:
        return

    args.model_types = [model_type]
    result = parse_results(args)
    df = result[0] if isinstance(result, tuple) else result
    if df is None or df.empty:
        return

    for scen in _get_scenarios(model_type):
        df_f = df[df[scen["fixed_param"]] == scen["fixed_value"]]
        for m in METRICS:
            _create_epoch_plot(
                df_f, scen["fixed_param"], scen["fixed_value"],
                scen["other_param"], cfg, m, out_dir, model_type, args.dataset,
            )
