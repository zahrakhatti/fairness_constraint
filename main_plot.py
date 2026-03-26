"""Main entry point for generating all plots.

Iterates over datasets and model types, producing:
  - cont1 plots (metric vs. constraint parameter)
  - cont2 plots (metric vs. epoch)
  - cont3 tables (constraint comparison)
"""
import os
from config import get_args, update_config
from cont1 import process_model_type
from cont2 import plot_model_type
from cont3 import create_constraint_comparison
from plot_config import get_output_dir

if __name__ == "__main__":
    datasets = ["dutch", "acsincome", "law"]
    data_plots = ["train", "test"]

    for dataset in datasets:
        print(f"\n Processing dataset: {dataset}")
        update_config({"dataset": dataset})
        args = get_args()

        for mt in args.model_types:
            # --- cont3: constraint comparison (model type 8) ---
            if mt == 8:
                out_dir = get_output_dir("output", dataset, True, args.scaled, "train_cont_3")
                os.makedirs(out_dir, exist_ok=True)
                create_constraint_comparison(args.dataset, mt, out_dir, args)

            # --- cont1 & cont2: per-parameter and epoch plots (types 2,3,4) ---
            if mt in (2, 3, 4):
                for data_plot in data_plots:
                    args.data_plot = data_plot

                    out1 = get_output_dir(
                        "output", args.dataset, args.full_batch, args.scaled,
                        str(mt), f"{data_plot}_cont_1",
                    )
                    os.makedirs(out1, exist_ok=True)
                    process_model_type(args.dataset, mt, args, out1)

                    out2 = get_output_dir(
                        "output", args.dataset, args.full_batch, args.scaled,
                        str(mt), f"{data_plot}_cont_2",
                    )
                    os.makedirs(out2, exist_ok=True)
                    plot_model_type(mt, args, out2)
