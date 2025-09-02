import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def check_results_exist(datasets, model_type, epsilons, lmbd, init_seeds):
    """
    Quickly check which result files exist without processing them.
    Returns a dictionary of existing results for efficient skipping.
    """
    existing_results = {}
    
    eps_str_map = {eps: f"{eps:.1e}" if eps != float('inf') else "inf" for eps in epsilons}
    lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
    
    for dataset in datasets:
        for eps in epsilons:
            for init_seed in init_seeds:
                key = (dataset, eps, init_seed)
                
                results_path = os.path.join(
                    "output", "seeds", f"seed_{init_seed}", 
                    dataset, "full_batch_True", str(model_type), "train_results",
                    f"eps_{eps_str_map[eps]}_lambda_{lmbd_str}.csv"
                )
                
                existing_results[key] = os.path.exists(results_path)
    
    return existing_results


def load_training_results(results_path):
    """Load training results from CSV file."""
    try:
        # Read CSV, skipping comment lines that start with #
        df = pd.read_csv(results_path, comment='#')
        return df
    except Exception as e:
        print(f"Error reading {results_path}: {e}")
        return None


def create_clean_plots(datasets, model_type, epsilons, lmbd, init_seeds, output_base_dir="output/plots"):
    """
    Create clean plots with mean lines and variance shaded regions showing:
    1. Overall accuracy across epochs
    2. Constraint violation measure across epochs
    """
    
    # Check existing results first
    print("Checking existing results...")
    existing_results = check_results_exist(datasets, model_type, epsilons, lmbd, init_seeds)
    
    total_expected = len(datasets) * len(epsilons) * len(init_seeds)
    total_existing = sum(existing_results.values())
    
    print(f"Found {total_existing}/{total_expected} existing result files")
    
    if total_existing == 0:
        print("No results found to plot!")
        return
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each dataset and epsilon combination
    for dataset in datasets:
        for eps in epsilons:
            print(f"\nProcessing plots for {dataset}, epsilon={eps}")
            
            # Collect data for this dataset/epsilon combination
            all_data = []
            
            for init_seed in init_seeds:
                key = (dataset, eps, init_seed)
                
                if not existing_results[key]:
                    print(f"  Skipping missing: seed_{init_seed}")
                    continue
                
                # Build path
                eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
                lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
                
                results_path = os.path.join(
                    "output", "seeds", f"seed_{init_seed}", 
                    dataset, "full_batch_True", str(model_type), "train_results",
                    f"eps_{eps_str}_lambda_{lmbd_str}.csv"
                )
                
                # Load data
                df = load_training_results(results_path)
                if df is not None:
                    df['init_seed'] = init_seed
                    all_data.append(df)
                    print(f"  Loaded: seed_{init_seed} ({len(df)} epochs)")
            
            if not all_data:
                print(f"  No valid data found for {dataset}, eps={eps}")
                continue
            
            # Create plots
            create_accuracy_and_constraint_plots(
                all_data, dataset, eps, model_type, lmbd, output_base_dir
            )


def create_accuracy_and_constraint_plots(all_data, dataset, eps, model_type, lmbd, output_base_dir):
    """
    Create two clean plots with mean lines and variance shaded regions:
    1. Overall accuracy across epochs
    2. Constraint violation measure across epochs
    """
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Combine all data and calculate statistics across seeds
    combined_data = compute_variance_statistics(all_data)
    
    if combined_data is None:
        print(f"  Could not compute statistics for {dataset}, eps={eps}")
        return
    
    epochs = combined_data['epochs']
    
    # Plot 1: Overall Accuracy with variance
    acc_mean = combined_data['acc_overall_mean']
    acc_std = combined_data['acc_overall_std']
    
    ax1.plot(epochs, acc_mean, 'o-', color='blue', 
             label=f'Mean (n={combined_data["n_seeds"]} seeds)', 
             markersize=4, linewidth=2)
    ax1.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, 
                     alpha=0.3, color='blue', label='±1 std')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Overall Accuracy (%)')
    ax1.set_title(f'Overall Accuracy - {dataset.title()}, ε={eps}')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Constraint Violation Measure with variance
    if 'constraint_mean' in combined_data:
        const_mean = combined_data['constraint_mean']
        const_std = combined_data['constraint_std']
        measure_name = combined_data['measure_name']
        
        ax2.plot(epochs, const_mean, 'o-', color='red', 
                 label=f'Mean (n={combined_data["n_seeds"]} seeds)', 
                 markersize=4, linewidth=2)
        ax2.fill_between(epochs, const_mean - const_std, const_mean + const_std, 
                         alpha=0.3, color='red', label='±1 std')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Constraint Violation Measure')
        ax2.set_title(f'{measure_name} - {dataset.title()}, ε={eps}')
        ax2.set_ylim(-1, 0.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No constraint measure found', 
                 transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title(f'Constraint Violation - {dataset.title()}, ε={eps}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
    lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
    
    plot_filename = f"training_plots_{dataset}_model_{model_type}_eps_{eps_str}_lambda_{lmbd_str}_variance.png"
    plot_path = os.path.join(output_base_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  Plot saved: {plot_filename}")
    
    return plot_path


def compute_variance_statistics(all_data):
    """
    Compute mean and standard deviation across seeds for each epoch.
    """
    if not all_data:
        return None
    
    # Find common epochs across all seeds
    all_epochs = [set(df['epoch'].values) for df in all_data]
    common_epochs = sorted(list(set.intersection(*all_epochs)))
    
    if not common_epochs:
        print("    Warning: No common epochs found across seeds")
        return None
    
    # Initialize arrays for statistics
    n_epochs = len(common_epochs)
    n_seeds = len(all_data)
    
    acc_overall_values = np.zeros((n_seeds, n_epochs))
    constraint_values = np.zeros((n_seeds, n_epochs))
    
    # Determine which constraint measure to use
    constraint_column = None
    measure_name = None
    
    for col_name, display_name in [
        ('const_v_measure_di', 'Constraint Violation (DI)'),
        ('const_v_measure_ei', 'Constraint Violation (EI)'),
        ('const_v_measure_cov_ind', 'Constraint Violation (Cov)')
    ]:
        if col_name in all_data[0].columns:
            constraint_column = col_name
            measure_name = display_name
            break
    
    # Extract values for each seed
    for seed_idx, df in enumerate(all_data):
        for epoch_idx, epoch in enumerate(common_epochs):
            epoch_data = df[df['epoch'] == epoch]
            if len(epoch_data) > 0:
                acc_overall_values[seed_idx, epoch_idx] = epoch_data['acc_overall'].iloc[0]
                if constraint_column:
                    constraint_values[seed_idx, epoch_idx] = epoch_data[constraint_column].iloc[0]
    
    # Compute statistics
    result = {
        'epochs': np.array(common_epochs),
        'n_seeds': n_seeds,
        'acc_overall_mean': np.mean(acc_overall_values, axis=0),
        'acc_overall_std': np.std(acc_overall_values, axis=0),
    }
    
    if constraint_column:
        result.update({
            'constraint_mean': np.mean(constraint_values, axis=0),
            'constraint_std': np.std(constraint_values, axis=0),
            'measure_name': measure_name
        })
    
    return result


def create_summary_plots(datasets, model_type, epsilons, lmbd, init_seeds, output_base_dir="output/plots"):
    """
    Create summary plots showing final results across different epsilon values with error bars.
    """
    
    print("\nCreating summary plots...")
    
    # Check existing results
    existing_results = check_results_exist(datasets, model_type, epsilons, lmbd, init_seeds)
    
    summary_data = []
    
    for dataset in datasets:
        for eps in epsilons:
            eps_results = []
            
            for init_seed in init_seeds:
                key = (dataset, eps, init_seed)
                
                if not existing_results[key]:
                    continue
                
                # Build path and load final results
                eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
                lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
                
                results_path = os.path.join(
                    "output", "seeds", f"seed_{init_seed}", 
                    dataset, "full_batch_True", str(model_type), "train_results",
                    f"eps_{eps_str}_lambda_{lmbd_str}.csv"
                )
                
                df = load_training_results(results_path)
                if df is not None and len(df) > 0:
                    final_result = df.iloc[-1]
                    eps_results.append({
                        'dataset': dataset,
                        'eps': eps,
                        'init_seed': init_seed,
                        'acc_overall': final_result['acc_overall'],
                        'const_v_measure': final_result.get('const_v_measure_di', 
                                                          final_result.get('const_v_measure_ei', 0))
                    })
            
            if eps_results:
                # Calculate mean and std for this epsilon
                acc_values = [r['acc_overall'] for r in eps_results]
                const_values = [r['const_v_measure'] for r in eps_results]
                
                summary_data.append({
                    'dataset': dataset,
                    'eps': eps,
                    'acc_mean': np.mean(acc_values),
                    'acc_std': np.std(acc_values),
                    'const_mean': np.mean(const_values),
                    'const_std': np.std(const_values),
                    'n_seeds': len(eps_results)
                })
    
    if not summary_data:
        print("No data available for summary plots")
        return
    
    # Create summary plots with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))
    
    for dataset_idx, dataset in enumerate(datasets):
        dataset_data = [d for d in summary_data if d['dataset'] == dataset]
        if not dataset_data:
            continue
        
        eps_vals = [d['eps'] for d in dataset_data]
        acc_means = [d['acc_mean'] for d in dataset_data]
        acc_stds = [d['acc_std'] for d in dataset_data]
        const_means = [d['const_mean'] for d in dataset_data]
        const_stds = [d['const_std'] for d in dataset_data]
        
        color = colors[dataset_idx]
        
        # Plot accuracy summary with error bars
        ax1.errorbar(eps_vals, acc_means, yerr=acc_stds, 
                     fmt='o-', label=dataset.title(), markersize=6, linewidth=2,
                     color=color, capsize=5, capthick=2)
        
        # Plot constraint violation summary with error bars
        ax2.errorbar(eps_vals, const_means, yerr=const_stds,
                     fmt='o-', label=dataset.title(), markersize=6, linewidth=2,
                     color=color, capsize=5, capthick=2)
    
    ax1.set_xlabel('Epsilon (ε)')
    ax1.set_ylabel('Mean Overall Accuracy (%) ± std')
    ax1.set_title('Accuracy vs Epsilon')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Epsilon (ε)')
    ax2.set_ylabel('Mean Constraint Violation ± std')
    ax2.set_title('Constraint Violation vs Epsilon')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save summary plot
    lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
    summary_filename = f"summary_plots_model_{model_type}_lambda_{lmbd_str}_variance.png"
    summary_path = os.path.join(output_base_dir, summary_filename)
    
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Summary plot saved: {summary_filename}")
    
    return summary_path


def main_plotting():
    """
    Main function to create all plots. 
    """
    
    # Configuration
    datasets = ["acsincome", "dutch", "law"]
    model_type = 2
    epsilons = [0.5, 0.6, 0.7, 0.8, 0.9]
    lmbd = 'inf'
    init_seeds = [42, 123, 456, 789, 999]
    
    print("="*80)
    print("CREATING TRAINING PLOTS WITH VARIANCE ANALYSIS")
    print("="*80)
    print(f"Datasets: {datasets}")
    print(f"Model Type: {model_type}")
    print(f"Epsilons: {epsilons}")
    print(f"Lambda: {lmbd}")
    print(f"Init Seeds: {init_seeds}")
    
    # Create individual training plots with variance
    create_clean_plots(datasets, model_type, epsilons, lmbd, init_seeds)
    
    # Create summary plots with error bars
    create_summary_plots(datasets, model_type, epsilons, lmbd, init_seeds)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETED!")
    print("="*80)
    print("Plots saved in: output/plots/")


if __name__ == "__main__":
    main_plotting()