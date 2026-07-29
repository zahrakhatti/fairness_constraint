from config import get_args, update_config
import os
from cont1 import process_model_type
from cont2 import plot_model_type
from cont3 import create_constraint_comparison

if __name__ == "__main__":
    datasets = ["dutch", "acsincome", "law"]
    data_plots = ["train", "test"]
    for dataset in datasets:
        print(f"\n Processing dataset: {dataset}")
        update_config({"dataset": dataset})
        args = get_args()

        for mt in args.model_types:
            if mt in [8]:
                if args.scaled == False:
                    out_dir = f"output/scaled_{args.scaled}/{dataset}/full_batch_True/train_cont_3"
                else:
                    out_dir = f"output/{dataset}/full_batch_True/train_cont_3"
                os.makedirs(out_dir, exist_ok=True)
                create_constraint_comparison(args.dataset, mt, out_dir, args)
            if mt in [2, 3, 4]:
                for data_plot in data_plots:
                    args.data_plot = data_plot
                    if args.scaled == False:
                        out_dir = f"output/scaled_{args.scaled}/{args.dataset}/full_batch_{args.full_batch}/{mt}/{data_plot}_cont_1"
                    else:
                        out_dir = f"output/{args.dataset}/full_batch_{args.full_batch}/{mt}/{data_plot}_cont_1"
                    os.makedirs(out_dir, exist_ok=True)
                    process_model_type(args.dataset, mt, args, out_dir)
                    
                    if args.scaled == False:
                        out_dir = f"output/scaled_{args.scaled}/{args.dataset}/full_batch_{args.full_batch}/{mt}/{data_plot}_cont_2"
                    else:
                        out_dir = f"output/{args.dataset}/full_batch_{args.full_batch}/{mt}/{data_plot}_cont_2"
                    os.makedirs(out_dir, exist_ok=True)
                    plot_model_type(mt, args, out_dir)

