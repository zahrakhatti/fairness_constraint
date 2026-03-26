"""Generate per-parameter metric plots (delta/epsilon sweeps at fixed lambda).

Each plot shows one or more metrics on the y-axis against the constraint
parameter on the x-axis, with data points annotated.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from config import get_args
from parse import extract_eps_lambda_from_filename, get_result_files
from plot_config import (
    MODEL_CONFIG, FONT_SIZES, FIGURE_SIZE, GRID_ALPHA, DPI,
    apply_plot_style, format_x_tick,
)

args = get_args()

# ---------------------------------------------------------------------------
# Plot definitions
# ---------------------------------------------------------------------------
# Each entry describes one type of figure.  ``model_types`` controls which
# model types produce this plot; the rest mirrors the old PLOT_CONFIGS list
# exactly so that the same filenames and content are produced.
PLOT_CONFIGS = [
    dict(
        metrics=["delta_model", "delta_measure"],
        y_label=r"$\hat{\delta}$",
        filename_format="delta_model_vs_measure_{model_name}",
        y_limits=[0, 1.1],
        model_types=[2, 3, 4],
    ),
    dict(
        metrics=["acc_overall"],
        y_label="accuracy (%)",
        filename_format="accuracy_{model_name}",
        y_limits=[0, 100],
        model_types=[2, 3, 4],
    ),
    dict(
        metrics=["const_v_measure_di"],
        y_label=r"$\bar{c}_{di}$",
        filename_format="di_measure_{model_name}",
        y_limits=[-0.7, 0.1],
        model_types=[2, 3],
    ),
    dict(
        metrics=["const_v_measure_cov_ind"],
        y_label=r"$\|\bar{c}_{cov}\| - \epsilon$",
        filename_format="di_measure_{model_name}",
        y_limits=[-0.7, 0.1],
        model_types=[4],
    ),
    dict(
        metrics=["const_v_model"],
        y_label=r"$c_{di}$",
        filename_format="di_model_{model_name}",
        y_limits=[-0.7, 0.1],
        model_types=[2, 3],
    ),
    dict(
        metrics=["const_v_model"],
        y_label=r"$\|c_{cov}\| - \epsilon$",
        filename_format="di_model_{model_name}",
        y_limits=[-0.7, 0.1],
        model_types=[4],
    ),
    dict(
        metrics=["loss"],
        y_label="loss",
        filename_format="loss_{model_name}",
        y_limits=[0.2, 0.5],
        model_types=[2, 3, 4],
    ),
]

DEFAULT_LAMBDA = "inf"

# Metric labels for the delta plots (two-line case)
_METRIC_LABELS = {
    "delta_model": r"$\phi(t)$",
    "delta_measure": r"$\hat{y}$",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _filter_files_by_lambda(result_files, lambda_value=None):
    """Keep only files whose lambda matches *lambda_value* (default ``inf``)."""
    if lambda_value is None:
        lambda_value = DEFAULT_LAMBDA
    out = []
    for fp in result_files:
        _, lmbd = extract_eps_lambda_from_filename(os.path.basename(fp))
        if lambda_value == "inf" and (lmbd == float("inf") or str(lmbd).lower() == "inf"):
            out.append(fp)
        elif str(lmbd) == str(lambda_value):
            out.append(fp)
    return out


def _extract_data(result_files, model_type):
    """Read CSV files and return a DataFrame with one row per parameter value."""
    param = MODEL_CONFIG[model_type]["constraint_variable"]["param_name"]
    rows = []
    for fp in result_files:
        df = pd.read_csv(fp, comment="#")
        if df.empty:
            continue
        last = df.iloc[-1].to_dict()
        eps, _ = extract_eps_lambda_from_filename(os.path.basename(fp))
        last[param] = eps
        rows.append(last)
    if not rows:
        return None
    plot_df = pd.DataFrame(rows)
    if plot_df[param].duplicated().any():
        plot_df = plot_df.groupby(param, as_index=False).mean()
    return plot_df.sort_values(param)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _annotate_point(x, y, offset_y, va):
    """Add a single numeric annotation to a data point."""
    plt.annotate(
        f"{y:.2f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, offset_y),
        ha="center",
        va=va,
        fontsize=FONT_SIZES["annotations"],
    )


def _create_metric_plot(df, model_type, plot_cfg, output_dir, dataset):
    """Render and save a single metric plot."""
    if df is None or df.empty:
        return

    conf = MODEL_CONFIG[model_type]
    model_name = conf["model_name_template"].format(dataset=dataset)
    symbol = conf["constraint_variable"]["symbol"]
    param = conf["constraint_variable"]["param_name"]

    # Filter to metrics that actually exist in the data
    metrics = [m for m in plot_cfg["metrics"] if m in df.columns and not df[m].isna().all()]
    if not metrics:
        return

    is_delta = any(m in ("delta_model", "delta_measure") for m in metrics)

    # Pre-compute "which line is higher" lookup for two-line delta plots
    higher_at_x = {}
    if is_delta and len(metrics) > 1:
        for _, row in df.iterrows():
            vals = {m: row[m] for m in metrics if pd.notna(row[m])}
            if vals:
                higher_at_x[row[param]] = max(vals, key=vals.get)

    apply_plot_style()
    plt.figure(figsize=FIGURE_SIZE)
    plt.title(model_name)

    x_vals = sorted(df[param].unique())
    use_log = model_type == 4 and all(v > 0 for v in x_vals)

    for idx, metric in enumerate(metrics):
        mask = ~df[metric].isna()
        x_data = df.loc[mask, param]
        y_data = df.loc[mask, metric]

        plt.plot(
            x_data, y_data,
            marker=conf["markers"][idx],
            linestyle=conf["line_styles"][idx],
            color=conf["colors"][idx],
            linewidth=2,
            label=_METRIC_LABELS.get(metric),
            markersize=8,
        )

        # Annotate each point
        for x, y in zip(x_data, y_data):
            if is_delta and len(metrics) > 1:
                above = higher_at_x.get(x) == metric
                _annotate_point(x, y, 10 if above else -15, "bottom" if above else "top")
            elif metric == "delta_model":
                _annotate_point(x, y, -15, "top")
            else:
                _annotate_point(x, y, 10, "bottom")

    # X-axis
    if use_log:
        plt.xscale("log")
    plt.xticks(x_vals, [format_x_tick(v) for v in x_vals], rotation=30)
    plt.minorticks_off()
    plt.grid(True, which="minor", alpha=GRID_ALPHA / 2)

    plt.xlabel(f"${symbol}$")
    plt.ylabel(plot_cfg["y_label"])

    if is_delta:
        plt.ylim(*plot_cfg["y_limits"])
        plt.legend(loc="lower right")
    elif plot_cfg["y_limits"]:
        plt.ylim(*plot_cfg["y_limits"])

    plt.grid(True, alpha=GRID_ALPHA)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.15)

    safe_name = conf["model_type_name"].replace(" ", "_")
    fname = f"{plot_cfg['filename_format'].format(model_name=safe_name)}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def process_model_type(dataset, model_type, args, output_dir):
    """Generate all metric plots for a given model type and dataset."""
    args.model_types = [model_type]
    all_files, _ = get_result_files(args)
    files = [f for f in all_files if f"/{model_type}/" in f]
    result_files = _filter_files_by_lambda(files, DEFAULT_LAMBDA)
    df = _extract_data(result_files, model_type)
    for cfg in PLOT_CONFIGS:
        if model_type in cfg["model_types"]:
            _create_metric_plot(df, model_type, cfg, output_dir, dataset)
