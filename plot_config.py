"""Shared plotting configuration and utilities.

Centralizes all plotting constants, model configs, and helper functions
used across cont1, cont2, cont3, and main_plot.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
FONT_SIZES = {
    "title": 30,
    "axes": 24,
    "tick_labels": 18,
    "legend": 18,
    "annotations": 16,
}

# Figure defaults
FIGURE_SIZE = (12, 8)
GRID_ALPHA = 0.3
DPI = 300

# ---------------------------------------------------------------------------
# Model configuration (used by cont1, cont2, and cont3)
# ---------------------------------------------------------------------------
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_MARKERS = ["o", "s", "^", "v", "D", "*"]
_LINE_STYLES = ["-", "--", "-.", ":", "-", "--"]

MODEL_CONFIG = {
    2: {
        "model_name_template": "{dataset}, smoothed-step",
        "model_type_name": "smoothed-step",
        "constraint_variable": {
            "name": "Delta",
            "symbol": r"\delta",
            "param_name": "Delta",
            "log_scale": False,
        },
        "fairness_column": "dp_measure",
        "colors": _COLORS,
        "markers": _MARKERS,
        "line_styles": _LINE_STYLES,
    },
    3: {
        "model_name_template": "{dataset}, sigmoid",
        "model_type_name": "sigmoid",
        "constraint_variable": {
            "name": "Delta",
            "symbol": r"\delta",
            "param_name": "Delta",
            "log_scale": False,
        },
        "fairness_column": "dp_measure",
        "colors": _COLORS,
        "markers": _MARKERS,
        "line_styles": _LINE_STYLES,
    },
    4: {
        "model_name_template": "{dataset}, covariance",
        "model_type_name": "covariance",
        "constraint_variable": {
            "name": "Epsilon",
            "symbol": r"\varepsilon",
            "param_name": "epsilon",
            "log_scale": "auto",
        },
        "fairness_column": "dp_measure",
        "colors": _COLORS,
        "markers": _MARKERS,
        "line_styles": _LINE_STYLES,
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_output_dir(base, dataset, full_batch, scaled, *subdirs):
    """Build an output directory path respecting the scaled flag.

    Examples
    --------
    >>> get_output_dir("output", "dutch", True, False, "2", "train_cont_1")
    'output/scaled_False/dutch/full_batch_True/2/train_cont_1'
    >>> get_output_dir("output", "dutch", True, True, "2", "train_cont_1")
    'output/dutch/full_batch_True/2/train_cont_1'
    """
    if scaled is False:
        parts = ["output", f"scaled_{scaled}", dataset, f"full_batch_{full_batch}"]
    else:
        parts = ["output", dataset, f"full_batch_{full_batch}"]
    parts.extend(subdirs)
    return os.path.join(*parts)


def apply_plot_style():
    """Apply the common matplotlib rc settings."""
    plt.rcParams.update(
        {
            "font.size": FONT_SIZES["axes"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axes"],
            "xtick.labelsize": FONT_SIZES["tick_labels"],
            "ytick.labelsize": FONT_SIZES["tick_labels"],
            "legend.fontsize": FONT_SIZES["legend"],
        }
    )


def format_x_tick(value):
    """Format x-axis tick: scientific notation for <0.1, one decimal otherwise."""
    if abs(value) < 0.1:
        return f"{value:.0e}"
    return f"{value:.1f}"


def format_value(val):
    """Format a parameter value for display in labels and filenames."""
    if isinstance(val, str) and val.lower() == "inf":
        return "\u221e"
    if isinstance(val, (int, float)):
        if val == 0:
            return "0"
        if val == 1.0e18:
            return "inf"
        if abs(val) < 1e-1:
            return f"{val:.1e}"
        return f"{val:.1f}"
    return val


def generate_epoch_colors(n):
    """Return *n* evenly-spaced colours from the tab20 colormap."""
    return plt.cm.tab20(np.linspace(0, 1, n))
