import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from config import get_args
# Import functions from parse module
from parse import extract_eps_lambda_from_filename, get_result_files, parse_results
args = get_args()
# ──────────── Helper for x-axis formatting ────────────
def custom_format_x(value):
    """Format x‐axis ticks: scientific for <0.01, decimal otherwise."""
    if abs(value) < 0.1:
        # e.g. 1e-04, 5e-04, 1e-03
        return f"{value:.0e}"
    else:
        # e.g. 0.005, 0.050, 0.500
        return f"{value:.1f}"

# Centralized Configuration
# -------------------------

# Increase font sizes throughout the code
FONT_SIZES = {
    'title': 30,
    'axes': 24,
    'tick_labels': 18,
    'legend': 18,
    'annotations': 16
}

# Model-specific configurations
MODEL_CONFIG = {
    2: {
        'model_name_template': '{dataset}, smoothed-step',
        'model_type_name': 'smoothed-step', 
        'constraint_variable': {
            'name': 'Delta',
            'symbol': '\\delta',
            'param_name': 'Delta',
            'log_scale': False
        },
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        'markers': ['o', 's', '^', 'v', 'D', '*'],
        'line_styles': ['-', '--', '-.', ':', '-', '--'],
    },
    3: {
        'model_name_template': '{dataset}, sigmoid',
        'model_type_name': 'sigmoid',
        'constraint_variable': {
            'name': 'Delta',
            'symbol': '\\delta',
            'param_name': 'Delta',
            'log_scale': False
        },
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        'markers': ['o', 's', '^', 'v', 'D', '*'],
        'line_styles': ['-', '--', '-.', ':', '-', '--'],
    },
    4: {
        'model_name_template': '{dataset}, covariance',
        'model_type_name': 'covariance',
        'constraint_variable': {
            'name': 'Epsilon',
            'symbol': '\\varepsilon',
            'param_name': 'epsilon',
            'log_scale': 'auto'
        },
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        'markers': ['o', 's', '^', 'v', 'D', '*'],
        'line_styles': ['-', '--', '-.', ':', '-', '--'],
    }
}

# Define plot configurations
PLOT_CONFIGS = [
    {
        'metrics': ['delta_model', 'delta_measure'],
        'title_format': "{model_name}",
        'y_label': r"$\hat{\delta}$",
        'filename_format': "delta_model_vs_measure_{model_name}",
        'y_limits': [0, 1.1],
        'model_types': [2, 3, 4]
    },
    {
        'metrics': ['acc_overall'],
        'title_format': "{model_name}",
        'y_label': "accuracy (%)",
        'filename_format': "accuracy_{model_name}",
        'y_limits': [0, 100],
        'model_types': [2, 3, 4]
    },
    {
        'metrics': ['const_v_measure_di'],
        'title_format': "{model_name}",
        'y_label': r"$\bar{c}_{di}$",
        'filename_format': "di_measure_{model_name}",
        'y_limits': [-0.7, 0.1],
        'model_types': [2, 3]
    },
    {
        'metrics': ['const_v_measure_cov_ind'],
        'title_format': "{model_name}",
        'y_label': r"$\|\bar{c}_{cov}\| - \epsilon$",
        'filename_format': "di_measure_{model_name}",
        'y_limits': [-0.7, 0.1],
        'model_types': [4]
    },
    {
        'metrics': ['const_v_model'],
        'title_format': "{model_name}",
        'y_label': r"$c_{di}$",
        'filename_format': "di_model_{model_name}",
        'y_limits': [-0.7, 0.1],
        'model_types': [2, 3]
    },
    {
        'metrics': ['const_v_model'],
        'title_format': "{model_name}",
        'y_label': r"$\|c_{cov}\| - \epsilon$",
        'filename_format': "di_model_{model_name}",
        'y_limits': [-0.7, 0.1],
        'model_types': [4]
    },
    {
        'metrics': ['loss'],
        'title_format': "{model_name}",
        'y_label': "loss",
        'filename_format': "loss_{model_name}",
        'y_limits': [0.2, 0.5],
        'model_types': [2, 3, 4]
    }
]

DEFAULT_LAMBDA = 'inf'
FIGURE_SIZE = (12, 8) 
GRID_ALPHA = 0.3
DPI = 300 

# ──────────── Utility functions (filtering & parsing) ────────────
def filter_files_by_lambda(result_files, lambda_value=None):
    if lambda_value is None:
        lambda_value = DEFAULT_LAMBDA
    filtered = []
    for fp in result_files:
        _, lmbd = extract_eps_lambda_from_filename(os.path.basename(fp))
        if (lambda_value == 'inf' and (lmbd == float('inf') or str(lmbd).lower() == 'inf')) \
           or str(lmbd) == str(lambda_value):
            filtered.append(fp)
    return filtered

def extract_data_for_plotting(result_files, model_type):
    rows = []
    for fp in result_files:
        df = pd.read_csv(fp, comment='#')
        if df.empty:
            continue
        last = df.iloc[-1].to_dict()
        eps, _ = extract_eps_lambda_from_filename(os.path.basename(fp))
        param = MODEL_CONFIG[model_type]['constraint_variable']['param_name']
        last[param] = eps
        rows.append(last)
    if not rows:
        return None
    plot_df = pd.DataFrame(rows)
    param = MODEL_CONFIG[model_type]['constraint_variable']['param_name']
    if plot_df[param].duplicated().any():
        plot_df = plot_df.groupby(param, as_index=False).mean()
    return plot_df.sort_values(param)

# ──────────── Main plotting function ────────────
def create_metric_plot(df, model_type, plot_config, output_dir, dataset):
    if df is None or df.empty:
        return

    conf = MODEL_CONFIG[model_type]
    # Use dataset in model name for plot titles
    model_name = conf['model_name_template'].format(dataset=dataset)
    model_type_name = conf['model_type_name']
    
    symbol     = conf['constraint_variable']['symbol']
    param_name = conf['constraint_variable']['param_name']
    metrics    = [m for m in plot_config['metrics']
                  if m in df.columns and not df[m].isna().all()]
    if not metrics:
        return

    is_delta = any(m in ['delta_model','delta_measure'] for m in metrics)

    plt.figure(figsize=FIGURE_SIZE)
    plt.rcParams.update({
        'font.size':       FONT_SIZES['axes'],
        'axes.titlesize':  FONT_SIZES['title'],
        'axes.labelsize':  FONT_SIZES['axes'],
        'xtick.labelsize': FONT_SIZES['tick_labels'],
        'ytick.labelsize': FONT_SIZES['tick_labels'],
        'legend.fontsize': FONT_SIZES['legend'],
    })
    # Use dataset-inclusive name for title
    plt.title(plot_config['title_format'].format(model_name=model_name))

    x_vals = sorted(df[param_name].unique())
    use_log = (model_type == 4 and all(v>0 for v in x_vals))

    # For delta plots with multiple metrics, we need to store data points
    if is_delta and len(metrics) > 1:
        data_by_x = {}
        for metric in metrics:
            mask = ~df[metric].isna()
            for _, row in df.loc[mask].iterrows():
                x_val = row[param_name]
                if x_val not in data_by_x:
                    data_by_x[x_val] = {}
                data_by_x[x_val][metric] = row[metric]

    # Plot each metric
    for idx, metric in enumerate(metrics):
        mask   = ~df[metric].isna()
        x_data = df.loc[mask, param_name]
        y_data = df.loc[mask, metric]

        if metric == 'delta_model':
            lbl = r"$\phi(t)$"
        elif metric == 'delta_measure':
            lbl = r"$\hat{y}$"
        else:
            lbl = None
        plt.plot(
            x_data, y_data,
            marker    = conf['markers'][idx],
            linestyle = conf['line_styles'][idx],
            color     = conf['colors'][idx],
            linewidth = 2,
            label     = lbl,
            markersize= 8
        )

        # Annotate data points with intelligent positioning
        for x, y in zip(x_data, y_data):
            # Determine annotation position
            if is_delta and len(metrics) > 1:
                # For delta plots with multiple lines, check if this is the highest line at this x
                is_highest = True
                for other_metric in metrics:
                    if other_metric != metric and x in data_by_x and other_metric in data_by_x[x]:
                        if data_by_x[x][other_metric] > y:
                            is_highest = False
                            break
                
                # Position based on whether this line is above or below others
                if is_highest:
                    xytext = (0, 10)  # Above for higher line
                    va = 'bottom'
                else:
                    xytext = (0, -15)  # Below for lower line
                    va = 'top'
            else:
                # For other plot types or single metric delta plots
                if metric == 'delta_model':
                    # Delta model annotations below the line
                    xytext = (0, -15)
                    va = 'top'
                else:
                    # Other metrics' annotations above the line
                    xytext = (0, 10)
                    va = 'bottom'
            
            # Add annotation with calculated position
            plt.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=xytext,
                ha='center',
                va=va,
                fontsize=FONT_SIZES['annotations']
            )
        
    # X-axis ticks & labels
    if use_log:
        plt.xscale('log')
    plt.xticks(
        x_vals,
        [custom_format_x(v) for v in x_vals],
        rotation=30
    )
    plt.minorticks_off()
    plt.grid(True, which='minor', alpha=GRID_ALPHA/2)

    plt.xlabel(f"${symbol}$")
    plt.ylabel(plot_config['y_label'])

    if is_delta:
        plt.ylim(*plot_config['y_limits']) 
        plt.legend(loc='lower right')
    elif plot_config['y_limits']:
        plt.ylim(*plot_config['y_limits'])

    plt.grid(True, alpha=GRID_ALPHA)
    
    # Force consistent axes positioning for all plots
    plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.15)

    # Use model_type_name for filenames to preserve original filename format
    safe_name = model_type_name.replace(' ', '_')
    fname = f"{plot_config['filename_format'].format(model_name=safe_name)}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=DPI, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Saved {fname}")

# ──────────── Processing pipeline ────────────
def process_model_type(dataset, model_type, args, output_dir):
    args.model_types = [model_type]
    all_files, _ = get_result_files(args)
    files = [f for f in all_files if f"/{model_type}/" in f]
    result_files = filter_files_by_lambda(files, DEFAULT_LAMBDA)
    df = extract_data_for_plotting(result_files, model_type)
    for cfg in PLOT_CONFIGS:
        if model_type in cfg['model_types']:
            create_metric_plot(df, model_type, cfg, output_dir, dataset)
