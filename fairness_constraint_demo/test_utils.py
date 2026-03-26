# test_utils.py
import os
import torch
from save_utils import ResultsManager

def format_eps_lambda_for_checkpoint(eps, lmbd):
    """
    Format eps and lambda consistently for checkpoint directory names.
    This should match exactly how they are saved during training.
    """
    eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
    lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
    return eps_str, lmbd_str

def format_eps_lambda_for_results(eps, lmbd):
    """
    Format eps and lambda for results file names (scientific notation).
    """
    eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
    lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
    return eps_str, lmbd_str

def evaluate_single_model_on_test(model_filepath, testloader, args):
    """
    Load a single saved model and evaluate on test dataset.
    Uses the EXACT SAME column names as training results for easy parsing.
    """
    from train_utils import test_acc
    
    results_manager = ResultsManager()
    
    # Load model
    model, model_parameters = results_manager.load_model(model_filepath, device=args.device)
    
    # If model was trained with DataParallel, wrap it again
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Evaluate on test set
    test_result = {}
    
    # Evaluate accuracy
    acc, acc_male, acc_female = test_acc(model, testloader, args, fairness_criteria='accdiff')
    test_result["acc_overall"] = acc
    test_result["acc_male"] = acc_male
    test_result["acc_female"] = acc_female

    # Evaluate fairness metrics
    dp_measure = test_acc(model, testloader, args, fairness_criteria='demographic_parity')
    test_result["dp_measure"] = dp_measure.item()

    covariance_ind_measure = test_acc(model, testloader, args, fairness_criteria='covariance_independence')
    c_cov_1 = covariance_ind_measure - args.epsilon
    c_cov_2 = - covariance_ind_measure - args.epsilon
    test_result["const_v_measure_cov_ind"] = max(c_cov_1, c_cov_2).item()

    c_bar_1, c_bar_2, DI_measure = test_acc(model, testloader, args, fairness_criteria='disparate_impact')
    test_result["const_v_measure_di"] = DI_measure.item()

    c_bar_1, c_bar_2, EI_measure = test_acc(model, testloader, args, fairness_criteria='equal_impact')
    test_result["const_v_measure_ei"] = EI_measure.item()

    # Check gender ratio
    pos_prob_s_0_measure, pos_prob_s_1_measure = test_acc(model, testloader, args, fairness_criteria='pos_pred_rate')
    test_result["measure_pos_prob_s_0"] = pos_prob_s_0_measure.item()
    test_result["measure_pos_prob_s_1"] = pos_prob_s_1_measure.item()
    
    delta1_measure = pos_prob_s_0_measure/(pos_prob_s_1_measure+1e-15)
    delta2_measure = pos_prob_s_1_measure/(pos_prob_s_0_measure+1e-15)
    test_result["delta_measure"] = min(delta1_measure, delta2_measure).item()
    
    return test_result


def evaluate_all_epoch_models_on_test(base_dir, eps, lmbd, testloader, args):
    """
    Load all saved epoch models and evaluate on test dataset.
    This creates test results in the same format as training results (epoch by epoch).
    """
    # Use consistent formatting for checkpoint directory
    eps_str, lmbd_str = format_eps_lambda_for_checkpoint(eps, lmbd)
    
    checkpoint_subdir = os.path.join(base_dir, "checkpoints", f"eps_{eps_str}_lambda_{lmbd_str}")
    
    print(f"Looking for checkpoint directory: {checkpoint_subdir}")
    
    if not os.path.exists(checkpoint_subdir):
        print(f"No checkpoint directory found: {checkpoint_subdir}")
        
        # Try alternative naming patterns as fallback
        alternative_patterns = [
            f"eps_{eps:.1e}_lambda_{lmbd_str}",  # Scientific notation
            f"eps_{float(eps)}_lambda_{lmbd_str}",  # Float format
        ]
        
        found_alternative = False
        for pattern in alternative_patterns:
            alt_dir = os.path.join(base_dir, "checkpoints", pattern)
            print(f"  Trying alternative: {alt_dir}")
            if os.path.exists(alt_dir):
                checkpoint_subdir = alt_dir
                found_alternative = True
                print(f"  Found alternative checkpoint directory: {checkpoint_subdir}")
                break
        
        if not found_alternative:
            print("No checkpoint directory found with any naming pattern")
            return []
    
    # Get all epoch model files
    epoch_files = []
    for file in os.listdir(checkpoint_subdir):
        if file.startswith('epoch_') and file.endswith('.pth'):
            epoch_num = int(file.split('epoch_')[1].split('.pth')[0])
            epoch_files.append((epoch_num, file))
    
    # Sort by epoch number
    epoch_files.sort(key=lambda x: x[0])
    
    if not epoch_files:
        print(f"No epoch models found in {checkpoint_subdir}")
        return []
    
    test_results_all_epochs = []
    
    print(f"Evaluating {len(epoch_files)} epoch models on test data...")
    
    for epoch_num, filename in epoch_files:
        model_filepath = os.path.join(checkpoint_subdir, filename)
        
        try:
            # Evaluate this epoch's model
            test_result = evaluate_single_model_on_test(model_filepath, testloader, args)
            
            # Add epoch number to the result
            test_result['epoch'] = epoch_num
            
            test_results_all_epochs.append(test_result)
            
            # Print progress every 10 epochs or for the last epoch
            if epoch_num % 10 == 0 or epoch_num == epoch_files[-1][0]:
                print(f"  Evaluated epoch {epoch_num} - Acc: {test_result.get('acc_overall', 0):.2f}%")
                
        except Exception as e:
            print(f"Error evaluating epoch {epoch_num}: {e}")
            continue
    
    print(f"Completed evaluation of {len(test_results_all_epochs)} epochs")
    return test_results_all_epochs


def get_results_filename(results_dir, eps, lmbd):
    """Generate filename for results based on epsilon and lambda values."""
    # Use scientific notation for results files
    eps_str, lmbd_str = format_eps_lambda_for_results(eps, lmbd)
    
    filename = f"eps_{eps_str}_lambda_{lmbd_str}.csv"
    return os.path.join(results_dir, filename)


def create_output_dirs(args, model_type):
    """Create output directories based on the specified directory structure."""
    if args.scaled == False:
        base_dir = os.path.join("output", f"scaled_{args.scaled}", args.dataset, f"full_batch_{args.full_batch}", str(model_type))
    else:
        base_dir = os.path.join("output", args.dataset, f"full_batch_{args.full_batch}", str(model_type))
    
    results_manager = ResultsManager()
    model_dir, train_results_dir, test_results_dir, checkpoints_dir, logs_dir = results_manager.create_directory_structure(base_dir)
        
    return base_dir, train_results_dir, test_results_dir, model_dir


def run_test_evaluation_after_training(args, model_type, eps, lmbd_val, testloader):
    """
    Run test evaluation for all epoch models after training is completed.
    This function focuses ONLY on test evaluation, no training metrics.
    """
    # Create output directories
    base_dir, train_results_dir, test_results_dir, model_dir = create_output_dirs(args, model_type)
    
    # Generate test results filename
    test_results_file = get_results_filename(test_results_dir, eps, lmbd_val)
    
    # Skip if test results already exist
    if os.path.exists(test_results_file):
        print(f"\nSkipping existing test results: {test_results_file}")
        return
    
    print(f"\n{'='*50}")
    print(f"RUNNING TEST EVALUATION FOR ALL EPOCHS")
    print(f"Dataset: {args.dataset}, Model: {model_type}, Eps: {eps}, Lambda: {lmbd_val}")
    print(f"{'='*50}")
    
    try:
        # Initialize results manager
        results_manager = ResultsManager()
        
        # Evaluate all epoch models on test data
        print("Evaluating all epoch models on test data...")
        test_results_all_epochs = evaluate_all_epoch_models_on_test(
            base_dir, eps, lmbd_val, testloader, args
        )
        
        if not test_results_all_epochs:
            print("No test results generated - no epoch models found")
            return
        
        # Get parameters for saving (same style as training file)
        parameters = {
            'dataset': args.dataset,
            'sensitive_feature': args.sf,
            'model_type': model_type,
            'epsilon': getattr(args, 'epsilon', 0),
            'lmbd': lmbd_val,
            'delta': getattr(args, 'delta', 0),
            'Total num. of epochs': len(test_results_all_epochs),
            'full_batch': args.full_batch,
            'batch_size': args.batch_size,
            'threshold': args.threshold,
            'init_seed': args.init_seed,
            'input_size': args.input_size,
            'hidden_sizes': args.hidden_sizes,
            'num_classes': args.num_classes,
            'upper_bound': getattr(args, 'upper_bound', []),
            'lower_bound': getattr(args, 'lower_bound', []),
            'eps_smooth': getattr(args, 'eps_smooth', 0.0001)
        }
        
        # Save test results in the same format as training results
        print("Saving test results for all epochs...")
        results_manager.save_results(test_results_all_epochs, parameters, test_results_file, result_type="test")
        
        # Get final epoch test results for summary
        final_test_result = test_results_all_epochs[-1] if test_results_all_epochs else {}
        
        # Print simple test summary
        print(f"\nTest Evaluation Summary:")
        print(f"  Epochs evaluated: {len(test_results_all_epochs)}")
        
        print(f"\nFinal Test Metrics (Epoch {final_test_result.get('epoch', 'N/A')}):")
        key_test_metrics = ['acc_overall', 'acc_male', 'acc_female', 
                          'dp_measure', 'const_v_measure_di', 'const_v_measure_ei', 'delta_measure']
        for key in key_test_metrics:
            if key in final_test_result:
                value = final_test_result[key]
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.5f}")
        
        print(f"\nTest results saved to: {test_results_file}")
        
    except Exception as e:
        print(f"Error during test evaluation: {e}")
        import traceback
        traceback.print_exc()