import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from config import get_args
from parse import parse_results

# Increased font sizes for better readability
FONT_SIZES = {
    'title': 30,
    'axes': 24,
    'tick_labels': 16,
    'legend': 14,
    'annotations': 12
}

# Function to generate a colorful palette using 'tab20' colormap
def generate_colors(num_colors):
    return plt.cm.tab20(np.linspace(0, 1, num_colors))

# Model configurations with lowercase names
MODEL_CONFIG = {
    2: {
        'model_name_template': '{dataset}, smoothed-step',  # Template with dataset placeholder
        'param_name': 'delta',
        'param_symbol': 'δ',
        'fairness_column': 'dp_measure',
        'colors': generate_colors(20),
        'line_styles': ['-', '--', '-.', ':', '-', '--'],
    },
    3: {
        'model_name_template': '{dataset}, sigmoid',  # Template with dataset placeholder
        'param_name': 'delta',
        'param_symbol': 'δ',
        'fairness_column': 'dp_measure',
        'colors': generate_colors(20),
        'line_styles': ['-', '--', '-.', ':', '-', '--'],
    },
    4: {
        'model_name_template': '{dataset}, covariance',  # Template with dataset placeholder
        'param_name': 'epsilon',
        'param_symbol': 'ε',
        'fairness_column': 'dp_measure',
        'colors': generate_colors(20),
        'line_styles': ['-', '--', '-.', ':', '-', '--'],
    }
}

# Metrics to plot
METRICS = [
    {
        'column': 'fairness_column',
        'title': "{dataset}",           
        'y_label': r"$\bar{r}_{dp}(w)$",              # use r_{dp}(w) for the fairness measure
        'filename': "demographic_parity_measure_{fixed_param}_{fixed_value}",
        'y_limits': None
    },
    {
        'column': 'acc_overall',
        'title': "{dataset}",          
        'y_label': "training accuracy (%)",           # lower‐case accuracy
        'filename': "accuracy_{fixed_param}_{fixed_value}",
        'y_limits': [0, 100]
    }
]

# Uniform figure size
FIGURE_SIZE = (12, 8)
GRID_ALPHA = 0.3

def format_value(val):
    if isinstance(val, str) and val.lower() == 'inf':
        return '∞'
    if isinstance(val, (int, float)):
        if val == 0:
            return '0'
        if val == 1.0e+18:
            return 'inf'
        if abs(val) < 1e-1:
            return f"{val:.1e}"
        return f"{val:.1f}"
    return val

def create_plot(df_filtered, fixed_param, fixed_value, other_param,
                model_config, metric_config, output_dir, model_type, dataset):
    if df_filtered.empty:
        print(f"No data for {fixed_param}={fixed_value}. Skipping plot.")
        return

    metric_column = metric_config['column']
    if metric_column == 'fairness_column':
        metric_column = model_config['fairness_column']

    if metric_column not in df_filtered.columns:
        print(f"Column {metric_column} not found in DataFrame.")
        return

    # Determine the symbol for the changing parameter based on model type
    if other_param == 'lmbd':
        param_symbol = 'λ'
    elif other_param == 'epsilon':
        if model_type == 4:
            param_symbol = 'ε'
        else:  # model_type == 2 or 3
            param_symbol = 'δ'

    # Set fonts
    plt.rcParams.update({
        'font.size': FONT_SIZES['axes'],
        'axes.titlesize': FONT_SIZES['title'],
        'axes.labelsize': FONT_SIZES['axes'],
        'xtick.labelsize': FONT_SIZES['tick_labels'],
        'ytick.labelsize': FONT_SIZES['tick_labels'],
        'legend.fontsize': FONT_SIZES['legend']
    })

    plt.figure(figsize=FIGURE_SIZE)


    unique_vals = sorted(df_filtered[other_param].unique())
    for i, val in enumerate(unique_vals):
        df_val = df_filtered[df_filtered[other_param] == val]
        if df_val.empty:
            continue

        style = model_config['line_styles'][i % len(model_config['line_styles'])]
        color = model_config['colors'][i % len(model_config['colors'])]

        plt.plot(df_val['epoch'],
                 df_val[metric_column],
                 label=f"{param_symbol} = {format_value(val)}",  # Use the dynamic symbol
                 linestyle=style,
                 color=color,
                 linewidth=2.5)

    # Format the model name with the dataset for the filename
    model_name = model_config['model_name_template'].format(dataset=dataset)
    
    # Title & labels
    plt.title(metric_config['title'].format(dataset=dataset))
    plt.xlabel("epoch")  # lowercase
    plt.ylabel(metric_config['y_label'])

    if metric_config['y_limits']:
        plt.ylim(*metric_config['y_limits'])

    # Legend in bottom‐right, no title
    plt.legend(loc='lower right', framealpha=0.5)

    plt.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    
    # Filename
    fixed_str = "inf" if fixed_value == float('inf') else format_value(fixed_value)
    fname = metric_config['filename'].format(
        fixed_param=fixed_param,
        fixed_value=fixed_str
    )
    
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, fname)}.png")

def plot_model_type(model_type, args, out_dir):
    cfg = MODEL_CONFIG.get(model_type)
    if not cfg:
        return

    args.model_types = [model_type]
    result = parse_results(args)
    df = result[0] if isinstance(result, tuple) else result
    if df is None or df.empty:
        return


    scenarios = (
        
        [{'fixed_param': 'epsilon', 'other_param': 'lmbd', 'fixed_value': 0},
         {'fixed_param': 'lmbd', 'other_param': 'epsilon', 'fixed_value': float('inf')}]
        if model_type != 4 else
        [{'fixed_param': 'epsilon', 'other_param': 'lmbd', 'fixed_value': 1.0e+18},
         {'fixed_param': 'lmbd', 'other_param': 'epsilon', 'fixed_value': float('inf')}]
    )

    for scen in scenarios:
        df_f = df[df[scen['fixed_param']] == scen['fixed_value']]
        for m in METRICS:
            create_plot(df_f, scen['fixed_param'], scen['fixed_value'],
                        scen['other_param'], cfg, m, out_dir, model_type, args.dataset)
