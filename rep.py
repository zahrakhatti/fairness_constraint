# rep.py

import os
import numpy as np
import torch
import random
import time
import pandas as pd
from train_utils import train_mlp
from test_utils import run_test_evaluation_after_training
from models import Feedforward
from dataloader import get_dataset
from config import get_args, update_config
from save_utils import ResultsManager
from resource_monitor import ResourceMonitor


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_output_dirs(args, model_type, init_seed):
    """Create output directories based on the specified directory structure with init_seed."""
    # Use ResultsManager with original structure
    original_base = os.path.join("output", args.dataset, f"full_batch_{args.full_batch}", str(model_type))
    results_manager = ResultsManager()
    model_dir, train_results_dir, test_results_dir, checkpoints_dir, logs_dir = results_manager.create_directory_structure(original_base)
    
    # Replace with seed structure
    seed_base = os.path.join("output", "seeds", f"seed_{init_seed}", args.dataset, f"full_batch_{args.full_batch}", str(model_type))
    seed_model_dir = model_dir.replace(original_base, seed_base)
    seed_train_dir = train_results_dir.replace(original_base, seed_base)
    seed_test_dir = test_results_dir.replace(original_base, seed_base)
    
    # Create directories
    for dir_path in [seed_model_dir, seed_train_dir, seed_test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return seed_base, seed_train_dir, seed_test_dir, seed_model_dir

def get_results_filename(results_dir, eps, lmbd):
    """Generate filename for results based on epsilon and lambda values."""
    eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
    lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
    filename = f"eps_{eps_str}_lambda_{lmbd_str}.csv"
    return os.path.join(results_dir, filename)


def get_model_filename(model_dir, eps, lmbd):
    """Generate filename for model based on epsilon and lambda values."""
    eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
    lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
    filename = f"eps_{eps_str}_lambda_{lmbd_str}.pth"
    return os.path.join(model_dir, filename)


def run_training_only(args, model_type, eps, lmbd, testloader, init_seed):
    """Run model training for a specific model type, epsilon, and lambda value - training only."""
    print(f"Using device: {args.device}")
    
    set_seed(init_seed)
    base_dir, train_results_dir, test_results_dir, model_dir = create_output_dirs(args, model_type, init_seed)
    
    if isinstance(lmbd, str):
        if lmbd.lower() == 'inf':
            lmbd_val = float('inf')
        else:
            lmbd_val = float(lmbd)
    else:
        lmbd_val = lmbd
    
    train_results_file = get_results_filename(train_results_dir, eps, lmbd_val)
    model_file = get_model_filename(model_dir, eps, lmbd_val)
    
    if os.path.exists(train_results_file) and os.path.exists(model_file):
        print(f"\nSkipping existing training results: {train_results_file} and {model_file}")
        return model_file, base_dir, eps, lmbd_val, None, train_results_file
    
    print(f"\n{'-'*50}")
    print(f"Training model type {model_type}, epsilon={eps}, lambda={lmbd}, init_seed={init_seed}")
    print(f"{'-'*50}")
    
    monitor = ResourceMonitor(monitor_interval=1.0)
    monitor.start_monitoring()
    training_start_time = time.time()
    
    try:
        set_seed(init_seed)
        args.checkpoint_base_dir = base_dir  # Set the seed-based directory
        model = Feedforward(args.input_size, args.hidden_sizes, args.num_classes).to(args.device)
        if args.device == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        trainloader, _, s_mean = get_dataset(args.dataset, args.sf, args.batch_size, model)
        print(f"Dataset '{args.dataset}' loaded with protected class: {args.sf}, batch size: {args.batch_size}")
        
        print("Starting model training...")
        model, train_results, parameters = train_mlp(model, model_type, trainloader, testloader, args)
        training_end_time = time.time()
        
        monitor.stop_monitoring()
        resource_summary = monitor.get_summary()
        
        results_manager = ResultsManager()
        
        print("\nSaving trained model...")
        model_filepath = results_manager.save_final_model(model, model_file, parameters)
        
        print("Saving training results...")
        results_manager.save_results(train_results, parameters, train_results_file, result_type="train")
        
        training_time_sec = training_end_time - training_start_time
        training_time_min = training_time_sec / 60
        
        final_train_result = train_results[-1] if train_results else {}
        
        training_metrics = {
            'init_seed': init_seed,
            'total_training_time_min': training_time_min,
            'total_epochs': len(train_results),
            'train_peak_memory_mb': resource_summary.get('peak_memory_mb', 0),
            'train_cpu_hours': resource_summary.get('cpu_hours', 0),
            'final_train_acc_overall': final_train_result.get('acc_overall', 0),
            'final_train_acc_male': final_train_result.get('acc_male', 0),
            'final_train_acc_female': final_train_result.get('acc_female', 0),
            'final_train_dp_measure': final_train_result.get('dp_measure', 0),
            'final_train_delta_measure': final_train_result.get('delta_measure', 0)
        }
        
        if torch.cuda.is_available():
            training_metrics['train_peak_gpu_memory_mb'] = resource_summary.get('peak_gpu_memory_mb', 0)
            training_metrics['train_avg_gpu_memory_mb'] = resource_summary.get('avg_gpu_memory_mb', 0)
        
        print("Logging training metrics...")
        results_manager.log_experiment_metrics(base_dir, model_type, eps, lmbd_val, training_metrics)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Seed: {init_seed}")
        print(f"Training time: {training_time_min:.2f} minutes")
        print(f"Total epochs: {len(train_results)}")
        
        monitor.cleanup()
        
        return model_filepath, base_dir, eps, lmbd_val, training_metrics, train_results_file
        
    except Exception as e:
        monitor.stop_monitoring()
        monitor.cleanup()
        
        print(f"Error during training for model type {model_type}, epsilon={eps}, lambda={lmbd}, init_seed={init_seed}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None


def run_test_evaluation(model_filepath, base_dir, eps, lmbd_val, testloader, args, model_type, training_metrics=None):
    """Run test evaluation using the test_utils module."""
    run_test_evaluation_after_training(args, model_type, eps, lmbd_val, testloader)


def get_final_epoch_results(results_path):
    """Get the final epoch results from a CSV file."""
    try:
        results_df = pd.read_csv(results_path, comment='#', engine='python')
        return results_df.iloc[-1]
    except Exception as e:
        print(f"Error reading {results_path}: {str(e)}")
        return None


def collect_results_across_seeds(datasets, model_type, eps, lmbd, init_seeds):
    """Collect final epoch results across all init_seeds for variance analysis."""
    all_results = []
    
    for dataset in datasets:
        dataset_results = []
        for init_seed in init_seeds:
            eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
            lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
            
            results_path = os.path.join(
                "output", "seeds", f"seed_{init_seed}", 
                dataset, "full_batch_True", str(model_type), "train_results",
                f"eps_{eps_str}_lambda_{lmbd_str}.csv"
            )
            
            print(f"Looking for: {results_path}")
            
            if os.path.exists(results_path):
                final_results = get_final_epoch_results(results_path)
                if final_results is not None:
                    result_dict = {
                        'dataset': dataset,
                        'init_seed': init_seed,
                        'acc_overall': final_results['acc_overall'],
                        'acc_male': final_results['acc_male'], 
                        'acc_female': final_results['acc_female'],
                        'dp_measure': final_results.get('dp_measure', 0),
                        'delta_measure': final_results.get('delta_measure', 0),
                    }
                    
                    if 'const_v_measure_di' in final_results:
                        result_dict['const_v_measure_di'] = final_results['const_v_measure_di']
                    if 'const_v_measure_ei' in final_results:
                        result_dict['const_v_measure_ei'] = final_results['const_v_measure_ei']
                    
                    dataset_results.append(result_dict)
                    all_results.append(result_dict)
                else:
                    print(f"Could not read results from {results_path}")
            else:
                print(f"Results file not found: {results_path}")
        
        if dataset_results:
            print(f"\nFound {len(dataset_results)} results for dataset {dataset}")
        else:
            print(f"\nNo results found for dataset {dataset}")
    
    return pd.DataFrame(all_results) if all_results else None


def create_publication_table(results_df, output_dir, model_type, eps, lmbd):
    """Create publication-ready table with mean ± std."""
    
    if results_df is None or len(results_df) == 0:
        print("No results to summarize!")
        return None
    
    try:
        from tabulate import tabulate
    except ImportError:
        print("Note: tabulate not available, will use basic formatting")
        tabulate = None
    
    column_mapping = {
        'acc_overall': 'Accuracy',
        'acc_male': 'Acc. Male', 
        'acc_female': 'Acc. Female',
        'dp_measure': 'Dem. Parity',
        'delta_measure': 'Eq. Odds'
    }
    
    available_metrics = [col for col in column_mapping.keys() if col in results_df.columns]
    
    summary_data = []
    datasets = sorted(results_df['dataset'].unique())
    
    for dataset in datasets:
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        if len(dataset_data) > 0:
            row = {'Dataset': dataset, 'n_init_seeds': len(dataset_data)}
            
            for metric in available_metrics:
                values = dataset_data[metric].values
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                row[column_mapping[metric]] = f"{mean_val:.3f} ± {std_val:.3f}"
            
            summary_data.append(row)
    
    display_columns = ['Dataset'] + [column_mapping[m] for m in available_metrics]
    display_df = pd.DataFrame([{col: row.get(col, 'N/A') for col in display_columns} for row in summary_data])
    
    summary_filename = f"publication_table_model_{model_type}_eps_{eps}_lambda_{lmbd}.csv"
    summary_path = os.path.join(output_dir, summary_filename)
    display_df.to_csv(summary_path, index=False)
    
    latex_filename = f"publication_table_model_{model_type}_eps_{eps}_lambda_{lmbd}.tex"
    latex_path = os.path.join(output_dir, latex_filename)
    
    latex_table = display_df.to_latex(index=False, escape=False, 
                                     column_format='l' + 'c' * (len(display_columns) - 1))
    
    with open(latex_path, 'w') as f:
        f.write("% Publication-ready table for ML paper\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Experimental results across 5 random init seeds. Values show mean ± standard deviation.}\n")
        f.write("\\label{tab:results}\n")
        f.write(latex_table)
        f.write("\\end{table}\n")
    
    print(f"\n{'='*80}")
    print("PUBLICATION-READY RESULTS TABLE")
    print("="*80)
    
    if tabulate:
        print(tabulate(display_df, headers='keys', tablefmt='grid'))
    else:
        print(display_df.to_string(index=False))
    
    print(f"\nFiles saved:")
    print(f"  CSV: {summary_path}")
    print(f"  LaTeX: {latex_path}")
    
    return display_df


def run_single_experiment(dataset, model_type, eps, lmbd, init_seed):
    """Run a single experiment with specified parameters and init_seed."""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: Dataset={dataset}, Model={model_type}, Eps={eps}, Lambda={lmbd}, Init_Seed={init_seed}")
    print(f"{'='*80}")
    
    update_config({"dataset": dataset})
    args = get_args()
    args.init_seed = init_seed
    
    set_seed(init_seed)
    model_temp = Feedforward(args.input_size, args.hidden_sizes, args.num_classes).to(args.device)
    _, testloader, _ = get_dataset(args.dataset, args.sf, args.batch_size, model_temp)
    
    set_seed(init_seed)
    args.lr = 0.5 if args.full_batch == True else 0.1
    args.lmbd = lmbd
    
    if model_type in [2, 3, 8, 9]:
        args.delta = eps
    if model_type in [4, 5]:
        args.epsilon = eps
    
    model_filepath, base_dir, eps_val, lmbd_val, training_metrics, train_results_file = run_training_only(
        args, model_type, eps, lmbd, testloader, init_seed
    )
    
    if model_filepath is not None:
        run_test_evaluation(
            model_filepath, base_dir, eps_val, lmbd_val, 
            testloader, args, model_type
        )
        print(f"✓ Experiment completed successfully for init_seed {init_seed}")
        return True
    else:
        print(f"✗ Experiment failed for init_seed {init_seed}")
        return False


if __name__ == "__main__":
    # Configuration
    init_seeds = [42, 123, 456, 789, 999]  # Use init_seed like your config
    datasets = ["dutch", "law"]
    model_type = 2
    epsilons = [0.5, 0.6, 0.7, 0.8, 0.9]
    lmbd = 'inf'
    
    print(f"Running experiments with {len(init_seeds)} different init seeds")
    print(f"Configuration: Datasets={datasets}, Model Type={model_type}, Epsilons={epsilons}, Lambda={lmbd}")
    print(f"Init Seeds: {init_seeds}")
    
    total_experiments = len(init_seeds) * len(datasets) * len(epsilons)
    completed_experiments = 0
    
    # Clean nested loop structure like main_train.py
    for dataset in datasets:
        print(f"\n{'='*100}")
        print(f"PROCESSING DATASET: {dataset}")
        print(f"{'='*100}")
        
        for eps in epsilons:
            print(f"\nProcessing epsilon: {eps}")
            
            for init_seed in init_seeds:
                print(f"\nInit seed {init_seed} for dataset {dataset}, eps {eps}")
                
                start_time = time.time()
                success = run_single_experiment(dataset, model_type, eps, lmbd, init_seed)
                elapsed_time = time.time() - start_time
                
                if success:
                    completed_experiments += 1
                    print(f"✓ Completed in {elapsed_time/60:.2f} minutes")
                
                progress = (completed_experiments / total_experiments) * 100
                print(f"Progress: {completed_experiments}/{total_experiments} ({progress:.1f}%)")
    
    print(f"\n{'='*100}")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*100}")
    
    # Analysis section
    print(f"\nAnalyzing variance across seeds...")
    
    # Analyze results for each epsilon value
    for eps in epsilons:
        print(f"\nAnalyzing results for epsilon={eps}")
        
        results_df = collect_results_across_seeds(datasets, model_type, eps, lmbd, init_seeds)
        
        if results_df is not None and len(results_df) > 0:
            analysis_dir = os.path.join("output", "publication_analysis", f"eps_{eps}")
            os.makedirs(analysis_dir, exist_ok=True)
            
            print(f"\nCreating publication-ready analysis for eps={eps}...")
            display_df = create_publication_table(results_df, analysis_dir, model_type, eps, lmbd)
            
            print(f"Results for eps={eps} saved in: {analysis_dir}")
        else:
            print(f"No results found for eps={eps}!")
    
    print(f"\n{'='*80}")
    print("PUBLICATION ANALYSIS COMPLETED!")
    print("="*80)
    print(f"Results saved in: output/publication_analysis/")
    
    print(f"\nFinal Summary:")
    print(f"  Total experiments: {completed_experiments}/{total_experiments}")
    print(f"  Results saved in: output/seeds/seed_{{init_seed}}/...")
    print(f"  Analysis saved in: output/publication_analysis/")