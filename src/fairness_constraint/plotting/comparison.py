"""Generate constraint-comparison tables (DI-only, EI-only, Both DI-EI).

Reads final-epoch results for model types 2, 8, and 9 and produces a
side-by-side CSV table per epsilon value.
"""
import os
import pandas as pd
from ..config import get_args
from .config import get_output_dir

args = get_args()

# Column mapping for extracting metrics from result CSVs
_COLUMN_NAMES = [
    "acc_overall", "acc_male", "acc_female",
    "const_v_model", "EI_model",
    "const_v_measure_di", "const_v_measure_ei",
]

_TABLE_COLUMNS = [
    "Overall Acc", "Male Acc", "Female Acc",
    "DI constraint Violation Model", "EI constraint Violation Model",
    "DI Constraint Violation Measure", "EI Constraint Violation Measure",
]


def _read_final_row(path):
    """Return the last row of a CSV (ignoring comment lines), or None."""
    try:
        df = pd.read_csv(path, comment="#", engine="python")
        return df.iloc[-1]
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def _safe(series, col, default=0.0):
    """Safely extract a value from a Series."""
    return series[col] if col in series.index else default


def _results_dir(dataset, model_type, scaled):
    """Build the train_results directory for a model type."""
    return get_output_dir("output", dataset, True, scaled, str(model_type), "train_results")


def create_constraint_comparison(dataset, model_type, output_dir, args):
    """Create comparison tables for epsilon values 0.8 and 0.9."""
    epsilon_values = ["8.0e-01", "9.0e-01"]
    tables = {}

    for eps in epsilon_values:
        scenarios = [
            ("Only DI (\u03b4={eps})",   eps,       "0.0e+00", 2),
            ("Only EI (\u03b4={eps})",   "0.0e+00", eps,       9),
            ("Both DI-EI (\u03b4={eps})", eps,       eps,       8),
        ]

        rows = []
        for desc_tmpl, eps_di, eps_ei, actual_mt in scenarios:
            desc = desc_tmpl.format(eps=eps)

            # Pick the right directory and filename
            if eps_di == "0.0e+00":
                base = _results_dir(dataset, 9, args.scaled)
                fname = f"eps_{eps_ei}_lambda_inf.csv"
            elif eps_ei == "0.0e+00":
                base = _results_dir(dataset, 2, args.scaled)
                fname = f"eps_{eps_di}_lambda_inf.csv"
            else:
                base = _results_dir(dataset, 8, args.scaled)
                fname = f"eps_{eps_di}_lambda_inf.csv"

            path = os.path.join(base, fname)
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue

            final = _read_final_row(path)
            if final is None:
                continue

            print(f"Processing {path} (Model Type {actual_mt})")
            print(f"Available columns: {final.index.tolist()}")

            row = {"Scenario": desc}
            for src, dst in zip(_COLUMN_NAMES, _TABLE_COLUMNS):
                row[dst] = _safe(final, src)
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows).round(4)
            csv_path = os.path.join(output_dir, f"constraint_comparison_eps_{eps}.csv")
            df.to_csv(csv_path, index=False)

            print(f"\nResults for \u03b4 = {eps}:")
            print("=" * 100)
            print(df.to_string(index=False))
            print(f"\nResults saved to: {csv_path}")

            tables[eps] = df

    return tables
