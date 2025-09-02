# Fixed version of your main script with consistent directory logic

import os
import numpy as np
import torch
import random
import time
from train_utils import train_mlp
from test_utils import run_test_evaluation_after_training
from models import Feedforward
from dataloader import get_dataset
from config import get_args, update_config
from save_utils import ResultsManager
from resource_monitor import ResourceMonitor


def create_output_dirs(args, model_type):
    """Create output directories based on the specified directory structure."""
    # FIXED: Consistent directory structure logic
    if args.scaled == False:
        base_dir = os.path.join("output", f"scaled_{args.scaled}", args.dataset, f"full_batch_{args.full_batch}", str(model_type))
    else:
        base_dir = os.path.join("output", args.dataset, f"full_batch_{args.full_batch}", str(model_type))
    
    # Create the directories using ResultsManager
    results_manager = ResultsManager()
    model_dir, train_results_dir, test_results_dir, checkpoints_dir, logs_dir = results_manager.create_directory_structure(base_dir)
    
    print(f"Created directory structure:")
    print(f"  Model directory: {model_dir}")
    print(f"  Train results directory: {train_results_dir}")
    print(f"  Test results directory: {test_results_dir}")
    print(f"  Checkpoints directory: {checkpoints_dir}")
    print(f"  Logs directory: {logs_dir}")
        
    return base_dir, train_results_dir, test_results_dir, model_dir

def get_results_filename(results_dir, eps, lmbd):
    """Generate filename for results based on epsilon and lambda values."""
    # Format epsilon and lambda values consistently
    eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
    lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
    
    # Create filename in the format: eps_{eps_str}_lambda_{lmbd_str}.csv
    filename = f"eps_{eps_str}_lambda_{lmbd_str}.csv"
    return os.path.join(results_dir, filename)

def get_model_filename(model_dir, eps, lmbd):
    """Generate filename for model based on epsilon and lambda values."""
    eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
    lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
    
    filename = f"eps_{eps_str}_lambda_{lmbd_str}.pth"
    return os.path.join(model_dir, filename)

def check_if_files_exist(train_results_dir, test_results_dir, model_dir, eps, lmbd_val):
    """
    Check if training and testing files already exist.
    Returns: (train_exists, test_exists, model_exists)
    """
    train_results_file = get_results_filename(train_results_dir, eps, lmbd_val)
    test_results_file = get_results_filename(test_results_dir, eps, lmbd_val)
    model_file = get_model_filename(model_dir, eps, lmbd_val)
    
    train_exists = os.path.exists(train_results_file)
    test_exists = os.path.exists(test_results_file)
    model_exists = os.path.exists(model_file)
    
    print(f"  Checking files for eps={eps}, lambda={lmbd_val}:")
    print(f"    Train file: {'EXISTS' if train_exists else 'MISSING'} - {train_results_file}")
    print(f"    Test file: {'EXISTS' if test_exists else 'MISSING'} - {test_results_file}")
    print(f"    Model file: {'EXISTS' if model_exists else 'MISSING'} - {model_file}")
    
    return train_exists, test_exists, model_exists

def run_training_only(args, model_type, eps, lmbd, testloader, train_results_dir, test_results_dir, model_dir):
    """Run model training for a specific model type, epsilon, and lambda value - training only."""
    print(f"Using device: {args.device}")
    

    
    # Check if lmbd is a string (for 'inf')
    if isinstance(lmbd, str):
        if lmbd.lower() == 'inf':
            lmbd_val = float('inf')
        else:
            lmbd_val = float(lmbd)
    else:
        lmbd_val = lmbd
    
    # Generate filenames for train results and model
    train_results_file = get_results_filename(train_results_dir, eps, lmbd_val)
    model_file = get_model_filename(model_dir, eps, lmbd_val)
    
    # Skip if training files already exist
    if os.path.exists(train_results_file) and os.path.exists(model_file):
        print(f"\nSkipping existing training results: {train_results_file} and {model_file}")
        return model_file, eps, lmbd_val, None
    
    print(f"\n{'-'*50}")
    print(f"Training model type {model_type}, epsilon={eps}, lambda={lmbd}")
    print(f"{'-'*50}")
    
    # Start resource monitoring for the ENTIRE training process
    monitor = ResourceMonitor(monitor_interval=1.0)  # Sample every 1 second
    monitor.start_monitoring()
    training_start_time = time.time()
    
    try:
        
        # Initialize model
        model = Feedforward(args.input_size, args.hidden_sizes, args.num_classes).to(args.device)
        if args.device == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        # Get dataset
        trainloader, _, s_mean = get_dataset(args.dataset, args.sf, args.batch_size, model)
        print(f"Dataset '{args.dataset}' loaded with protected class: {args.sf}, batch size: {args.batch_size}")
        
        # Train model (now returns only model, train_results, parameters)
        print("Starting model training...")
        model, train_results, parameters = train_mlp(model, model_type, trainloader, testloader, args)
        training_end_time = time.time()
        
        # Stop resource monitoring immediately after training
        monitor.stop_monitoring()
        resource_summary = monitor.get_summary()
        
        # Initialize results manager
        results_manager = ResultsManager()
        
        # Save the trained model
        print("\nSaving trained model...")
        model_filepath = results_manager.save_final_model(model, model_file, parameters)
        
        # Save train results
        print("Saving training results...")
        results_manager.save_results(train_results, parameters, train_results_file, result_type="train")
        
        # Calculate timing metrics
        training_time_sec = training_end_time - training_start_time
        training_time_min = training_time_sec / 60
        
        # Get final training metrics
        final_train_result = train_results[-1] if train_results else {}
        
        # Prepare ONLY the training metrics for logging
        training_metrics = {
            'total_training_time_min': training_time_min,
            'total_epochs': len(train_results),
            'train_peak_memory_mb': resource_summary.get('peak_memory_mb', 0),
            'train_cpu_hours': resource_summary.get('cpu_hours', 0),
            # Add final training accuracy and fairness metrics
            'final_train_acc_overall': final_train_result.get('acc_overall', 0),
            'final_train_acc_male': final_train_result.get('acc_male', 0),
            'final_train_acc_female': final_train_result.get('acc_female', 0),
            'final_train_dp_measure': final_train_result.get('dp_measure', 0),
            'final_train_delta_measure': final_train_result.get('delta_measure', 0)
        }
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            training_metrics['train_peak_gpu_memory_mb'] = resource_summary.get('peak_gpu_memory_mb', 0)
            training_metrics['train_avg_gpu_memory_mb'] = resource_summary.get('avg_gpu_memory_mb', 0)
        
        # Create base_dir for logging
        base_dir = os.path.dirname(train_results_dir)  # Go up one level from train_results
        
        # Immediately log training metrics to the experiment CSV
        print("Logging training metrics...")
        results_manager.log_experiment_metrics(base_dir, model_type, eps, lmbd_val, training_metrics)
        
        # Print training summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print(f"\nTiming Summary:")
        print(f"  Training time: {training_time_min:.2f} minutes")
        print(f"  Total epochs: {len(train_results)}")
        
        print(f"\nResource Usage:")
        print(f"  Peak memory: {resource_summary.get('peak_memory_mb', 0):.1f} MB")
        print(f"  Average CPU: {resource_summary.get('avg_cpu_percent', 0):.1f}%")
        print(f"  Estimated CPU hours: {resource_summary.get('cpu_hours', 0):.3f}")
        if torch.cuda.is_available():
            print(f"  Peak GPU memory: {resource_summary.get('peak_gpu_memory_mb', 0):.1f} MB")

        print(f"\nFinal Training Metrics:")
        key_train_metrics = ['acc_overall', 'acc_male', 'acc_female', 
                           'dp_measure', 'const_v_measure_di', 'const_v_measure_ei', 'delta_measure']
        for key in key_train_metrics:
            if key in final_train_result:
                value = final_train_result[key]
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.5f}")
        
        print(f"\nFiles Created:")
        print(f"  Model: {model_filepath}")
        print(f"  Training results: {train_results_file}")
        print(f"  Training metrics logged to: {base_dir}/model_{model_type}_experiments.csv")
        
        # Cleanup
        monitor.cleanup()
        
        return model_filepath, eps, lmbd_val, training_metrics
        
    except Exception as e:
        # Stop monitoring even if there's an error
        monitor.stop_monitoring()
        monitor.cleanup()
        
        print(f"Error during training for model type {model_type}, epsilon={eps}, lambda={lmbd}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def run_test_evaluation(model_filepath, eps, lmbd_val, testloader, args, model_type):
    """Run test evaluation using the test_utils module."""
    # Call the test_utils function
    run_test_evaluation_after_training(args, model_type, eps, lmbd_val, testloader)

if __name__ == "__main__":
    # Get command line arguments
    datasets = ["dutch", "law", "acsincome"]

    # Track overall progress
    total_experiments = 0
    completed_experiments = 0
    
    for dataset in datasets:
        print(f"\n Processing dataset: {dataset}")
        update_config({"dataset": dataset})
        args = get_args()

        # Get test data once per dataset
        model_temp = Feedforward(args.input_size, args.hidden_sizes, args.num_classes).to(args.device)
        _, testloader, _ = get_dataset(args.dataset, args.sf, args.batch_size, model_temp)

        for model_type in args.model_types:
            print(f"\n Processing model type: {model_type}")
            
            # Create output directories once per model_type
            base_dir, train_results_dir, test_results_dir, model_dir = create_output_dirs(args, model_type)
            
            for scenario in range(2):
                if args.scaled == False:
                    scenario = 0
                if model_type in [2, 3]:
                    if scenario == 0:
                        lmbd_values = ['inf']
                        eps_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    if scenario == 1:
                        lmbd_values = [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
                        eps_values = [0]

                if model_type in [4, 5]:
                    if scenario == 0:
                        lmbd_values = ['inf']
                        eps_values = [1e-1, 5e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4]
                    if scenario == 1:
                        lmbd_values = [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
                        eps_values = [1e+18]

                if model_type in [8, 9]:
                    eps_values = [0.8, 0.9] 
                    lmbd_values = ['inf']

                total_experiments += len(eps_values) * len(lmbd_values)

                for eps in eps_values:
                    for lmbd in lmbd_values:
                        print(f"\n{'='*80}")
                        print(f"PROCESSING: Dataset={dataset}, Model={model_type}, Eps={eps}, Lambda={lmbd}")
                        print(f"{'='*80}")
                        
                        args.lr = 0.5 if args.full_batch == True else 0.1
                        
                        # Set the appropriate parameters based on model type
                        args.lmbd = lmbd
                        if model_type in [2, 3, 8, 9]:
                            args.delta = eps
                        if model_type in [4, 5]:
                            args.epsilon = eps

                        # Convert lmbd to proper value for checking
                        if isinstance(lmbd, str):
                            if lmbd.lower() == 'inf':
                                lmbd_val = float('inf')
                            else:
                                lmbd_val = float(lmbd)
                        else:
                            lmbd_val = lmbd

                        # Check what files already exist
                        train_exists, test_exists, model_exists = check_if_files_exist(
                            train_results_dir, test_results_dir, model_dir, eps, lmbd_val
                        )
                        
                        # Decision logic
                        if train_exists and model_exists and test_exists:
                            print("ALL FILES EXIST - SKIPPING COMPLETELY")
                            completed_experiments += 1
                            continue
                        
                        # Handle training
                        if train_exists and model_exists:
                            print("Training files exist - SKIPPING TRAINING")
                            model_filepath = get_model_filename(model_dir, eps, lmbd_val)
                        else:
                            print("Training files missing - RUNNING TRAINING")
                            model_filepath, eps_val, lmbd_val, training_metrics = run_training_only(
                                args, model_type, eps, lmbd, testloader, 
                                train_results_dir, test_results_dir, model_dir
                            )
                            
                            if model_filepath is None:
                                print("Training failed - skipping test evaluation")
                                completed_experiments += 1
                                continue
                        
                        # Handle testing
                        if test_exists:
                            print("Test files exist - SKIPPING TESTING")
                        else:
                            print("Test files missing - RUNNING TESTING")
                            run_test_evaluation(
                                model_filepath, eps, lmbd_val, 
                                testloader, args, model_type
                            )
                        
                        completed_experiments += 1
                        
                        # Print progress
                        progress = (completed_experiments / total_experiments) * 100
                        print(f"PROGRESS: {completed_experiments}/{total_experiments} ({progress:.1f}%)")

    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"Total processed: {completed_experiments}/{total_experiments}")