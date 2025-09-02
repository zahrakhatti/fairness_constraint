# train_utils script
import random
import torch
import os
import time
import numpy as np
from fairness_models import *
from fairness_matrics import *
from stochasticsqp import StochasticSQP
import logging
from save_utils import ResultsManager


def evaluate_test_dataset_from_saved_model(model_filepath, testloader, args):

    from save_utils import ResultsManager
    
    results_manager = ResultsManager()
    
    # Load model
    model, model_parameters = results_manager.load_model(model_filepath, device=args.device)
    
    # If model was trained with DataParallel, wrap it again
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Evaluate on test set
    test_result = {}
    
    # Evaluate test accuracy
    acc, acc_male, acc_female = test_acc(model, testloader, args, fairness_criteria='accdiff')
    test_result["test_acc_overall"] = acc
    test_result["test_acc_male"] = acc_male
    test_result["test_acc_female"] = acc_female

    # Evaluate fairness metrics on test set
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


def evaluate_test_dataset_from_saved_model_all_epochs(base_dir, eps, lmbd, testloader, args):
    """
    Load all saved epoch models and evaluate on test dataset.
    This is the original function from your code - kept as is for backward compatibility.
    """
    from save_utils import ResultsManager
    
    # Format eps and lambda for directory name
    eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
    lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
    
    checkpoint_subdir = os.path.join(base_dir, "checkpoints", f"eps_{eps}_lambda_{lmbd}")
    
    if not os.path.exists(checkpoint_subdir):
        print(f"No checkpoint directory found: {checkpoint_subdir}")
        return []
    
    # Get all epoch model files
    epoch_files = []
    for file in os.listdir(checkpoint_subdir):
        if file.startswith('epoch_') and file.endswith('.pth'):
            epoch_num = int(file.split('epoch_')[1].split('.pth')[0])
            epoch_files.append((epoch_num, file))
    
    # Sort by epoch number
    epoch_files.sort(key=lambda x: x[0])
    
    test_results_all_epochs = []
    results_manager = ResultsManager()
    
    print(f"Evaluating {len(epoch_files)} epoch models on test data...")
    
    for epoch_num, filename in epoch_files:
        model_filepath = os.path.join(checkpoint_subdir, filename)
        
        try:
            # Load model
            model, model_parameters = results_manager.load_model(model_filepath, device=args.device)
            
            # If model was trained with DataParallel, wrap it again
            if args.device == "cuda" and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            
            # Evaluate on test set
            test_result = evaluate_test_dataset_from_saved_model(model_filepath, testloader, args)
            test_result['epoch'] = epoch_num
            
            test_results_all_epochs.append(test_result)
            
            if epoch_num % 10 == 0 or epoch_num == epoch_files[-1][0]:
                print(f"  Evaluated epoch {epoch_num}")
                
        except Exception as e:
            print(f"Error evaluating epoch {epoch_num}: {e}")
            continue
    
    return test_results_all_epochs


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def log_info(model_type, args):
    # Create 'logs' directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    log_filename = os.path.join(log_dir, f"training_log_{model_type}_{time.strftime('%Y%m%d_%H%M')}.log")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # To display logs in the console as well
        ]
    )

def process_tensors(*tensors, device):
    """Process multiple tensors by reshaping and moving to specified device."""
    return [tensor.reshape(-1).to(device) for tensor in tensors]


# ----------------------------- Training Gradients and Jacobian -----------------------------
def compute_gradients(model, optimizer, loss, n_parameters, args):
    g = torch.zeros(n_parameters).to(args.device)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    
    # Add memory cleanup after gradient computation
    i = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_l = len(param.grad.view(-1))
            g[i:i+grad_l] = param.grad.view(-1).clone()  # Clone to preserve gradient
            param.grad = None  # Clear the gradient after copying
            i += grad_l
    
    torch.cuda.empty_cache()
    return g


def compute_jacobian(model, optimizer, c, n_constrs, n_parameters, args):
    """
    Compute and return the Jacobian for constraints.
    """
    J = torch.zeros(n_constrs, n_parameters).to(args.device) 
    for k in range(n_constrs):
        optimizer.zero_grad()
        c[k].backward(retain_graph=True)
        grads = torch.Tensor().to(args.device)
        for name, param in model.named_parameters():
            grads = torch.cat((grads, param.grad.view(-1).to(args.device)), 0) 
        J[k, :] = grads
    return J

# ----------------------------- Training Logic -----------------------------
def train_mlp(model, model_type, trainloader, testloader, args):
    print(f'model_type: {model_type}')
    torch.cuda.empty_cache()
    log_info(model_type, args)
    # Set random seed for reproducibility
    set_seed(1)
    # Initialize results manager for checkpointing
    results_manager = ResultsManager()

    # Add checkpoint saving frequency (save every 50 epochs by default)
    checkpoint_frequency = getattr(args, 'checkpoint_frequency', 1)


    # List to store results for each epoch
    results = []
    parameters = {
        'dataset': args.dataset,
        'sensitive_feature': args.sf,
        'model_type': model_type,
        'epsilon': args.epsilon,
        'lmbd': args.lmbd,
        'Total num. of epochs': args.epochs,
        'full_batch': args.full_batch,
        'batch_size': args.batch_size,
        'threshold': args.threshold,
        'init_seed': args.init_seed,
        'input_size': args.input_size,
        'hidden_sizes': args.hidden_sizes,
        'num_classes': args.num_classes,
        'upper_bound': args.upper_bound,
        'lower_bound': args.lower_bound,
        'delta': args.delta
        # 'checkpoint_frequency': checkpoint_frequency

    }

    n_parameters = sum(p.numel() for p in model.parameters())
    
    # Configure constraints based on model type
    if model_type in [2, 3, 9]:  # Two constraints (Disparate Impact)
        n_constrs = 2
        # For model types 2 and 3, we're treating both constraints like 
        # c_1 ≤ 0 and c_2 ≤ 0
        upper_bound = [0.0, 0.0]
        lower_bound = [1e+18, 1e+18]  # Effectively no lower bound
    elif model_type == 4:  # Covariance_dp with symmetric bounds
        n_constrs = 1
        # For covariance, the constraint is -epsilon ≤ c ≤ epsilon
        upper_bound = [args.epsilon]
        lower_bound = [args.epsilon]
    elif model_type in [5, 6, 7]:  # One constraint models with upper bound only
        n_constrs = 1
        # For these models, the constraint is c ≤ 0
        upper_bound = [args.epsilon]
        lower_bound = [args.epsilon] 

    elif model_type == 8:  # Combined model with DP and EO
        n_constrs = 4
        # For model type 8, all constraints are of the form c ≤ 0
        upper_bound = [0.0, 0.0, 0.0, 0.0]
        lower_bound = [1e+18, 1e+18, 1e+18, 1e+18]  # No lower bounds
    else:
        # Default (unconstrained)
        n_constrs = 0
        upper_bound = []
        lower_bound = []

    # Create optimizer with appropriate constraints
    optimizer = StochasticSQP(
        model.parameters(),
        lr=args.lr,
        epsilon=args.epsilon,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        n_parameters=n_parameters,
        n_constrs=n_constrs,
        merit_param_init=1,
        ratio_param_init=1,
    )

    # Training parameters
    criterion = torch.nn.BCELoss()
    
    # Add model-specific parameters

    parameters['eps_smooth'] = args.eps_smooth

    # Training loop variables
    iter_since_last_lr_adjustment = 0
    avg_merit = 0
    iter = 0

    # ====== Setup data for constraints based on full_batch ======
    if args.full_batch:
        # When full_batch is True, constraint_inputs will be set to current batch later
        constraint_inputs = None
        constraint_labels = None
        constraint_groups = None
        N_0 = None
        N_1 = None
    else:
        # When full_batch is False, extract fixed constraint data once
        all_inputs = []
        all_labels = []
        all_groups = []
        for inputs, labels, groups in trainloader:
            all_inputs.append(inputs)
            all_labels.append(labels)
            all_groups.append(groups)
        
        all_inputs = torch.cat(all_inputs).float().to(args.device)
        all_labels = torch.cat(all_labels).float().to(args.device)
        all_groups = torch.cat(all_groups).float().to(args.device)
        
        # Sample half of the data for constraints
        dataset_size = len(all_inputs)
        constraint_size = dataset_size // 3
        constraint_indices = torch.randperm(dataset_size)[:constraint_size]
        set_seed(args.init_seed)
        constraint_inputs = all_inputs[constraint_indices]
        constraint_labels = all_labels[constraint_indices]
        constraint_groups = all_groups[constraint_indices]
        
        # Calculate fixed N_0 and N_1 for regularization from the full dataset
        N_0 = (all_groups == 0).sum().float()
        N_1 = (all_groups == 1).sum().float()
                                             
    # ====== End of setup data for constraints based on full_batch ======
    # Epoch loop
    for epoch in range(args.epochs):
        iter += 1

        # Initialize current result for this epoch
        current_result = {'epoch': epoch + 1}
        current_result["lr"] = args.lr
        
        # Initialize c and J
        c = torch.tensor([]).to(args.device)

        # Iteration loop
        for i, (inputs, labels, groups) in enumerate(trainloader):
            N_0 = (groups == 0).sum().float()
            N_1 = (groups == 1).sum().float()
            inputs = inputs.float().to(args.device)
            labels = labels.float().to(args.device)
            groups = groups.float().to(args.device)
            # ====== Set constraint data based on full_batch flag ======
            if args.full_batch:
                # Use current batch for constraints when full_batch is True
                constraint_inputs = inputs
                constraint_labels = labels
                constraint_groups = groups
            # ===== End of setting constraint data based on full_batch flag ======
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss_init = loss.clone()
            current_result["loss"] = loss_init.item()

            # Regularization factor
            if args.lmbd == 'inf':
                lam = 0
            else:
                lam = 1/args.lmbd
            
            # Compute constraints based on model type
            if model_type in [2]:
                sum_s0, sum_s1, model_pos_prob_s_0, model_pos_prob_s_1, c_1, c_2 = probability_DI(model, constraint_inputs, constraint_groups, args.eps_smooth, args.delta, args)
                # model_pos_prob_s_0, model_pos_prob_s_1, _, _ = probability_DI(model, inputs, groups, args.eps_smooth, args.delta)
                sum_s0 = sum_s0.reshape(-1).to(args.device)
                sum_s1 = sum_s1.reshape(-1).to(args.device)
                model_pos_prob_s_0 = model_pos_prob_s_0.reshape(-1).to(args.device)
                model_pos_prob_s_1 = model_pos_prob_s_1.reshape(-1).to(args.device)
                c_1 = c_1.reshape(-1).to(args.device)
                c_2 = c_2.reshape(-1).to(args.device)

                # ====== Calculate regularization using fixed ratios ======
                if args.full_batch == False:
                    dp = sum_s0 / N_0 - sum_s1 / N_1
                else:
                    dp = model_pos_prob_s_0 - model_pos_prob_s_1
                # ====== End of calculate regularization using fixed ratios ======

                reg = torch.norm(dp, p=2) * torch.norm(dp, p=2)
                loss += lam * reg
                c = torch.cat((c_1, c_2), dim=0)
                
                # Store metrics
                current_result["const_v_model"] = torch.max(c_1, c_2).item()
                current_result["model_pos_prob_s_0"] = model_pos_prob_s_0.item()
                current_result["model_pos_prob_s_1"] = model_pos_prob_s_1.item()
                current_result["dp_model"] = abs(dp.item())
                delta1 = model_pos_prob_s_0/(model_pos_prob_s_1+1e-15)
                delta2 = model_pos_prob_s_1/(model_pos_prob_s_0+1e-15)
                current_result["delta_model"] = torch.min(delta1, delta2).item()

                _, _, ei_c_1, ei_c_2 = probability_EI(model, constraint_inputs, constraint_labels, constraint_groups, args.eps_smooth, args.delta, args)
                ei_c_1 = ei_c_1.reshape(-1).to(args.device)
                ei_c_2 = ei_c_2.reshape(-1).to(args.device)
                current_result["EI_model"] = torch.max(ei_c_1, ei_c_2).item()

            # Compute constraints based on model type
            if model_type in [3]:
                _, _, c_1, c_2 = probability_sigmoid(model, constraint_inputs, constraint_groups, args.delta, args)
                model_pos_prob_s_0, model_pos_prob_s_1, _, _ = probability_sigmoid(model, inputs, groups, args.delta, args)

                model_pos_prob_s_0 = model_pos_prob_s_0.reshape(-1).to(args.device)
                model_pos_prob_s_1 = model_pos_prob_s_1.reshape(-1).to(args.device)
                c_1 = c_1.reshape(-1).to(args.device)
                c_2 = c_2.reshape(-1).to(args.device)

                # ====== Calculate regularization using fixed ratios ======
                if args.full_batch == False:
                    # Convert probabilities back to sums by multiplying by the batch size
                    batch_size_s0 = (groups == 0).sum().float()
                    batch_size_s1 = (groups == 1).sum().float()
                    
                    sum_s0 = model_pos_prob_s_0 * batch_size_s0
                    sum_s1 = model_pos_prob_s_1 * batch_size_s1
                    
                    # Now calculate using fixed N_0 and N_1
                    dp = sum_s0 / N_0 - sum_s1 / N_1
                else:
                    dp = model_pos_prob_s_0 - model_pos_prob_s_1
                # ====== End of calculate regularization using fixed ratios ======

                reg = torch.norm(dp, p=2) * torch.norm(dp, p=2)
                loss += lam * reg
                c = torch.cat((c_1, c_2), dim=0)
                
                # Store metrics
                current_result["const_v_model"] = torch.max(c_1, c_2).item()
                current_result["model_pos_prob_s_0"] = model_pos_prob_s_0.item()
                current_result["model_pos_prob_s_1"] = model_pos_prob_s_1.item()
                current_result["dp_model"] = abs(dp.item())

                delta1 = model_pos_prob_s_0/(model_pos_prob_s_1+1e-15)
                delta2 = model_pos_prob_s_1/(model_pos_prob_s_0+1e-15)
                current_result["delta_model"] = torch.min(delta1, delta2).item()

            if model_type in [4]:  # Covariance_dp
                cov = covariance(model, constraint_inputs, constraint_groups).reshape(-1).to(args.device)
                cov_reg = covariance(model, inputs, groups).reshape(-1).to(args.device)
                reg = torch.norm(cov_reg, p=2) * torch.norm(cov_reg, p=2)
                loss += lam * reg
                c = cov
                
                c_1 = cov - args.epsilon
                c_2 = - cov - args.epsilon
                current_result["const_v_model"] = torch.max(c_1, c_2).item()
                
                
            if model_type in [5]:  # Probability_dp
                dp = probability_sigmoid_dp(model, constraint_inputs, constraint_groups, args.eps_smooth, args).reshape(-1).to(args.device)
                reg = torch.norm(dp, p=2) * torch.norm(dp, p=2)
                loss += lam * reg
                c = dp
                
                current_result["dp_model"] = dp.item()
                cov_dp_save = covariance(model, constraint_inputs, constraint_groups).reshape(-1).to(args.device)
                current_result["cov_dp_model"] = cov_dp_save.item()
                
            if model_type in [6]:  # covariance_eo
                cov_eo = covariance_eo(model, constraint_inputs, labels, constraint_groups).reshape(-1).to(args.device)
                reg = torch.norm(cov_eo, p=2) * torch.norm(cov_eo, p=2)
                loss += lam * reg
                c = cov_eo
                
                current_result["cov_eo_model"] = cov_eo.item()
                eo_save = probability_eo(model, constraint_inputs, labels, constraint_groups, args.eps_smooth_eo, args.pos_slope).reshape(-1).to(args.device)
                current_result["eo_model"] = eo_save.item()
                
            if model_type in [7]:  # Probability_eo
                eo = probability_eo(model, constraint_inputs, labels, constraint_groups, args.eps_smooth_eo, args.pos_slope).reshape(-1).to(args.device)
                reg = torch.norm(eo, p=2) * torch.norm(eo, p=2)
                loss += lam * reg
                c = eo
                
                current_result["eo_model"] = eo.item()
                cov_eo_save = covariance_eo(model, constraint_inputs, labels, constraint_groups).reshape(-1).to(args.device)
                current_result["cov_eo_model"] = cov_eo_save.item()
                

            if model_type in [8]:  # Probability_dp and Probability_eo
                # Get raw tensors from functions
                _, _, di_model_pos_prob_s_0, di_model_pos_prob_s_1, di_c_1, di_c_2 = probability_DI(model, constraint_inputs, constraint_groups, args.eps_smooth, args.delta, args)
                ei_model_pos_prob_s_0, ei_model_pos_prob_s_1, ei_c_1, ei_c_2 = probability_EI(model, constraint_inputs, labels, constraint_groups, args.eps_smooth, args.delta, args)

                ei_model_pos_prob_s_0, ei_model_pos_prob_s_1, ei_c_1, ei_c_2 = process_tensors(ei_model_pos_prob_s_0, ei_model_pos_prob_s_1, ei_c_1, ei_c_2, device=args.device)
                _, _, di_model_pos_prob_s_0, di_model_pos_prob_s_1, di_c_1, di_c_2 = process_tensors(_, _, di_model_pos_prob_s_0, di_model_pos_prob_s_1, di_c_1, di_c_2, device=args.device)


                c = torch.cat((di_c_1, di_c_2, ei_c_1, ei_c_2), dim=0)
                
                # Store metrics
                # Store DI metrics
                current_result["DI_model_c1"] = di_c_1.item()
                current_result["DI_model_c2"] = di_c_2.item()
                current_result["const_v_model"] = torch.max(di_c_1, di_c_2).item()
                current_result["DI_model_pos_prob_s_0"] = di_model_pos_prob_s_0.item()
                current_result["DI_model_pos_prob_s_1"] = di_model_pos_prob_s_1.item()

                # Calculate DI deltas
                delta1_di = di_model_pos_prob_s_0/(di_model_pos_prob_s_1+1e-15)
                delta2_di = di_model_pos_prob_s_1/(di_model_pos_prob_s_0+1e-15)
                current_result["delta_DI_model"] = torch.min(delta1_di, delta2_di).item()

                # Store EI metrics
                current_result["EI_model_c1"] = ei_c_1.item()
                current_result["EI_model_c2"] = ei_c_2.item()
                current_result["EI_model"] = torch.max(ei_c_1, ei_c_2).item()
                current_result["EI_model_pos_prob_s_0"] = ei_model_pos_prob_s_0.item()
                current_result["EI_model_pos_prob_s_1"] = ei_model_pos_prob_s_1.item()

                # Calculate EI deltas
                delta1_ei = ei_model_pos_prob_s_0/(ei_model_pos_prob_s_1+1e-15)
                delta2_ei = ei_model_pos_prob_s_1/(ei_model_pos_prob_s_0+1e-15)
                current_result["delta_EI_model"] = torch.min(delta1_ei, delta2_ei).item()

            if model_type in [9]:
                ei_model_pos_prob_s_0, ei_model_pos_prob_s_1, ei_c_1, ei_c_2 = probability_EI(model, constraint_inputs, labels, constraint_groups, args.eps_smooth, args.delta, args)
                ei_model_pos_prob_s_0, ei_model_pos_prob_s_1, ei_c_1, ei_c_2 = process_tensors(ei_model_pos_prob_s_0, ei_model_pos_prob_s_1, ei_c_1, ei_c_2, device=args.device)

                DI = ei_model_pos_prob_s_0 - ei_model_pos_prob_s_1
                reg = torch.norm(DI, p=2) * torch.norm(DI, p=2)
                loss += lam * reg
                c = torch.cat((ei_c_1, ei_c_2), dim=0)
                # Store EI metrics
                current_result["EI_model_c1"] = ei_c_1.item()
                current_result["EI_model_c2"] = ei_c_2.item()
                current_result["EI_model"] = torch.max(ei_c_1, ei_c_2).item()
                current_result["EI_model_pos_prob_s_0"] = ei_model_pos_prob_s_0.item()
                current_result["EI_model_pos_prob_s_1"] = ei_model_pos_prob_s_1.item()

                # Calculate EI deltas
                delta1_ei = ei_model_pos_prob_s_0/(ei_model_pos_prob_s_1+1e-15)
                delta2_ei = ei_model_pos_prob_s_1/(ei_model_pos_prob_s_0+1e-15)
                current_result["delta_EI_model"] = torch.min(delta1_ei, delta2_ei).item()

                _, _, _, _, c_1, c_2 = probability_DI(model, constraint_inputs, constraint_groups, args.eps_smooth, args.delta, args)
                current_result["const_v_model"] = torch.max(c_1, c_2).item()

            # Compute gradients
            g = compute_gradients(model, optimizer, loss, n_parameters, args).to(args.device)
            
            # Only compute Jacobian if there are constraints
            if n_constrs > 0:
                J = compute_jacobian(model, optimizer, c, n_constrs, n_parameters, args).to(args.device)
                optimizer.state['J'] = J
                optimizer.state['c'] = c.data
            
            optimizer.state['g'] = g
            optimizer.state['f'] = loss.data
            
            # Take optimization step
            optimizer.step()
            optimizer.zero_grad()
            
            # Record step information
            d = optimizer.state['d']
            current_result["g|2"] = torch.norm(g).item()
            
            if n_constrs > 0:
                current_result["J|2"] = torch.norm(J).item()
            
            current_result["d|2"] = torch.norm(d).item()
            
            # Get active constraints
            active_constraints = optimizer.state.get('active_constraints', [])
            current_result["active_constraints"] = len(active_constraints)



            # Compute merit function for LR adjustment
            tau = 1e-2
            eta = 0.85
            
            def compute_merit_function(tau, loss, c, epsilon):
                penalty = torch.tensor(0.0).to(args.device)
                                
                if model_type in [2, 3, 8, 9]:
                    penalty = torch.sum(torch.clamp(c, min=0))
                elif model_type == 4:  # Symmetric constraint
                    penalty = torch.clamp(torch.norm(c, p=1) - args.epsilon, min=0)
                else:  # Upper bound constraints
                    penalty = torch.sum(torch.clamp(c, min=0))
                    
                return tau * loss + penalty
            
            # Compute merit function and update learning rate if needed
            merit_f_value = compute_merit_function(tau, loss.item(), c, args.epsilon).item()
            current_result["merit_value"] = merit_f_value
            
            if iter == 0:
                avg_merit = merit_f_value
            else:
                avg_merit = eta * merit_f_value + (1 - eta) * avg_merit
            
            # Adaptive learning rate adjustment
            if iter >= 200 and args.lr >= 1e-7:
                if avg_merit <= merit_f_value:
                    if iter_since_last_lr_adjustment >= 5:
                        args.lr = args.lr/10
                        # Update optimizer learning rate
                        optimizer.step_size = args.lr
                        optimizer.step_size_init = args.lr
                        
                        logging.info(
                            f"Epoch {epoch + 1}: Adjusted learning rate to {args.lr} due to fluctuation in merit function improvement."
                        )
                        iter_since_last_lr_adjustment = 0
            
            iter_since_last_lr_adjustment += 1

        # End of epoch processing
        current_result["loss + reg"] = loss.item()
        
        # Evaluate fairness metrics
        acc, acc_male, acc_female = train_acc(model, trainloader, args, fairness_criteria='accdiff')
        current_result["acc_overall"] = acc
        current_result["acc_male"] = acc_male
        current_result["acc_female"] = acc_female


        dp_measure = train_acc(model, trainloader, args, fairness_criteria='demographic_parity')
        current_result["dp_measure"] = dp_measure.item()

        
        covariance_ind_measure = train_acc(model, trainloader, args, fairness_criteria='covariance_independence')
        c_cov_1 = covariance_ind_measure - args.epsilon
        c_cov_2 = - covariance_ind_measure - args.epsilon
        current_result["const_v_measure_cov_ind"] = max(c_cov_1, c_cov_2).item()


        c_bar_1, c_bar_2, DI_measure = train_acc(model, trainloader, args, fairness_criteria='disparate_impact')
        current_result["const_v_measure_di"] = DI_measure.item()

        c_bar_1, c_bar_2, EI_measure = train_acc(model, trainloader, args, fairness_criteria='equal_impact')
        current_result["const_v_measure_ei"] = EI_measure.item()

        # Check gender ratio
        pos_prob_s_0_measure, pos_prob_s_1_measure = train_acc(model, trainloader, args, fairness_criteria='pos_pred_rate')
        current_result["measure_pos_prob_s_0"] = pos_prob_s_0_measure.item()
        current_result["measure_pos_prob_s_1"] = pos_prob_s_1_measure.item()
        
        delta1_measure = pos_prob_s_0_measure/(pos_prob_s_1_measure+1e-15)
        delta2_measure = pos_prob_s_1_measure/(pos_prob_s_0_measure+1e-15)
        current_result["delta_measure"] = min(delta1_measure, delta2_measure).item()

        # Add current epoch results to all results
        results.append(current_result.copy())


        try:
            # Check if args has checkpoint_base_dir, otherwise use original logic
            if hasattr(args, 'checkpoint_base_dir') and args.checkpoint_base_dir is not None:
                base_dir = args.checkpoint_base_dir
            else:
                # Original directory structure
                if args.scaled == False:
                    base_dir = os.path.join("output", f"scaled_{args.scaled}", args.dataset, f"full_batch_{args.full_batch}", str(model_type))
                else:
                    base_dir = os.path.join("output", args.dataset, f"full_batch_{args.full_batch}", str(model_type))
            
            if model_type in [2, 3, 8, 9]:
                eps = getattr(args, 'delta', )
            if model_type in [4, 5]:
                eps = getattr(args, 'epsilon', )
            lmbd = args.lmbd
            
            results_manager.save_epoch_model(
                model, epoch + 1, base_dir, eps, lmbd, parameters
            )
                
        except Exception as e:
            logging.warning(f"Failed to save epoch model at epoch {epoch + 1}: {e}")



        # Print log information
        if (epoch % 1 == 0):
            if model_type in [1]:
                logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Train acc: {acc:.5f}, Loss: {loss_init:.5f}')
            elif model_type in [2]:
                ac_info = f", Active constraints: {len(active_constraints)}" if n_constrs > 0 else ""
                logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Train acc: {acc:.5f}, Loss: {loss_init.item():.5f}, Loss + Reg: {loss.item():.5f}, DI_c1: {c_1.item():.5f}, DI_c2: {c_2.item():.5f}, DI_measure: {DI_measure.item():.5f}, delta_model:{torch.min(delta1, delta2).item():.5f}, delta_measure:{min(delta1_measure, delta2_measure).item():.5f}')
            elif model_type in [3]:
                ac_info = f", Active constraints: {len(active_constraints)}" if n_constrs > 0 else ""
                logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Train acc: {acc:.5f}, Loss: {loss_init.item():.5f}, Loss + Reg: {loss.item():.5f}, c_1: {c_1.item():.5f}, c_2: {c_2.item():.5f}, DI_measure: {DI_measure.item():.5f}, delta_model:{torch.min(delta1, delta2).item():.5f}, delta_measure:{min(delta1_measure, delta2_measure).item():.5f}')
            elif model_type in [4]:
                ac_info = f", Active constraints: {len(active_constraints)}" if n_constrs > 0 else ""
                logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Train acc: {acc:.5f}, Loss: {loss_init.item():.5f}, Loss + Reg: {loss.item():.5f}, cov_dp: {c.item():.5f}, cov_dp_measure: {max(c_cov_1, c_cov_2).item():.5f}, dp_measure: {dp_measure.item():.5f}{ac_info}, delta_measure:{min(delta1_measure, delta2_measure).item():.5f}')
            elif model_type in [5]:
                ac_info = f", Active constraints: {len(active_constraints)}" if n_constrs > 0 else ""
                logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Train acc: {acc:.5f}, Loss: {loss_init.item():.5f}, Loss + Reg: {loss.item():.5f}, dp: {c.item():.5f}, cov_dp_measure: {max(c_cov_1, c_cov_2).item():.5f}, dp_measure: {dp_measure.item():.5f}{ac_info}')
            elif model_type in [6]:
                ac_info = f", Active constraints: {len(active_constraints)}" if n_constrs > 0 else ""
                logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Train acc: {acc:.5f}, Loss: {loss_init.item():.5f}, Loss + Reg: {loss.item():.5f}, cov_eo: {c.item():.5f}, cov_eo_measure: {max(c_cov_1, c_cov_2).item():.5f}{ac_info}')
            elif model_type in [7]:
                ac_info = f", Active constraints: {len(active_constraints)}" if n_constrs > 0 else ""
                logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Train acc: {acc:.5f}, Loss: {loss_init.item():.5f}, Loss + Reg: {loss.item():.5f}, eo: {c.item():.5f}, Cov_eo_measure: {max(c_cov_1, c_cov_2).item():.5f}{ac_info}')
            elif model_type in [8]:
                ac_info = f", Active constraints: {len(active_constraints)}" if n_constrs > 0 else ""
                logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Train acc: {acc:.5f}, Loss: {loss_init.item():.5f}, Loss + Reg: {loss.item():.5f}, DI: {c[0].item():.5f}, EI: {c[2].item():.5f}, di_measure: {DI_measure.item():.5f}, ei_measure: {EI_measure.item():.5f}{ac_info}')
            elif model_type in [9]:
                ac_info = f", Active constraints: {len(active_constraints)}" if n_constrs > 0 else ""
                logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Train acc: {acc:.5f}, Loss: {loss_init.item():.5f}, Loss + Reg: {loss.item():.5f}, EI_model: {torch.max(ei_c_1, ei_c_2).item():.5f}, ei_measure: {EI_measure.item():.5f}{ac_info}')# Add these functions to train_utils.py after the existing test_acc function

    logging.info(f'Training Accuracy: {acc:.2f}%, male accuracy: {acc_male}, female accuracy: {acc_female}')



    return model, results, parameters



def train_acc(model, trainloader, args, fairness_criteria='accdiff'):
    label_arr, pred_arr, group_arr = [], [], []
    threshold = 0.5

    with torch.no_grad():
        for j, data in enumerate(trainloader):
            inputs, labels, groups = data
            # Move data to device
            inputs = inputs.float().to(args.device)
            labels = labels.float().to(args.device)
            groups = groups.float().to(args.device)

            outputs = model(inputs)
            # Use device-aware tensor creation
            predicted = torch.where(outputs > threshold, 
                                  torch.ones_like(outputs, device=args.device), 
                                  torch.zeros_like(outputs, device=args.device))
            
            pred_arr.extend(predicted.cpu().detach().numpy())
            label_arr.extend(labels.view(-1).cpu().detach().numpy())
            group_arr.extend(groups.cpu().detach().numpy())

    label_arr = np.array(label_arr)
    pred_arr = np.array(pred_arr)
    group_arr = np.array(group_arr)

    if fairness_criteria == 'accdiff':
        accuracy, accuracy_male, accuracy_female = acc_diff_binary(label_arr, pred_arr, group_arr)
        return accuracy*100, accuracy_male*100, accuracy_female*100
    elif fairness_criteria == 'covariance_independence':
        covariance_ind_measure = Covariance_ind_binary(pred_arr, group_arr)
        return covariance_ind_measure
    elif fairness_criteria == 'demographic_parity':
        demographic_parity_measure = demographic_parity_binary(pred_arr, group_arr)
        return demographic_parity_measure
    elif fairness_criteria == 'Covariance_eo':
        covariance_eo_measure = Covariance_eo_binary(label_arr, pred_arr, group_arr)
        return covariance_eo_measure
    elif fairness_criteria == 'equal_opportunity':
        equal_opportunity_measure = equal_opportunity_binary(label_arr, pred_arr, group_arr)
        return equal_opportunity_measure
    elif fairness_criteria == 'disparate_impact':
        c_bar_1, c_bar_2, DI_measure = disparate_impact_binary(pred_arr, group_arr, args.delta)
        return c_bar_1, c_bar_2, DI_measure
    elif fairness_criteria == 'equal_impact':
        c_bar_1, c_bar_2, EI_measure = equal_impact_binary(label_arr, pred_arr, group_arr, args.delta)
        return c_bar_1, c_bar_2, EI_measure
    elif fairness_criteria == 'pos_pred_rate':
        positive_prob_s_0, positive_prob_s_1 = positive_prediction_ratio(pred_arr, group_arr)
        return positive_prob_s_0, positive_prob_s_1


def test_acc(model, testloader, args, fairness_criteria='accdiff'):
    label_arr, pred_arr, group_arr = [], [], []
    threshold = 0.5

    with torch.no_grad():
        for j, data in enumerate(testloader):
            inputs, labels, groups = data
            # Move data to device
            inputs = inputs.float().to(args.device)
            labels = labels.float().to(args.device)
            groups = groups.float().to(args.device)

            outputs = model(inputs)
            # Use tensor creation
            predicted = torch.where(outputs > threshold, 
                                  torch.ones_like(outputs, device=args.device), 
                                  torch.zeros_like(outputs, device=args.device))
            
            pred_arr.extend(predicted.cpu().detach().numpy())
            label_arr.extend(labels.view(-1).cpu().detach().numpy())
            group_arr.extend(groups.cpu().detach().numpy())

    label_arr = np.array(label_arr)
    pred_arr = np.array(pred_arr)
    group_arr = np.array(group_arr)

    if fairness_criteria == 'accdiff':
        accuracy, accuracy_male, accuracy_female = acc_diff_binary(label_arr, pred_arr, group_arr)
        return accuracy*100, accuracy_male*100, accuracy_female*100
    elif fairness_criteria == 'covariance_independence':
        covariance_ind_measure = Covariance_ind_binary(pred_arr, group_arr)
        return covariance_ind_measure
    elif fairness_criteria == 'demographic_parity':
        demographic_parity_measure = demographic_parity_binary(pred_arr, group_arr)
        return demographic_parity_measure
    elif fairness_criteria == 'Covariance_eo':
        covariance_eo_measure = Covariance_eo_binary(label_arr, pred_arr, group_arr)
        return covariance_eo_measure
    elif fairness_criteria == 'equal_opportunity':
        equal_opportunity_measure = equal_opportunity_binary(label_arr, pred_arr, group_arr)
        return equal_opportunity_measure
    elif fairness_criteria == 'disparate_impact':
        c_bar_1, c_bar_2, DI_measure = disparate_impact_binary(pred_arr, group_arr, args.delta)
        return c_bar_1, c_bar_2, DI_measure
    elif fairness_criteria == 'equal_impact':
        c_bar_1, c_bar_2, EI_measure = equal_impact_binary(label_arr, pred_arr, group_arr, args.delta)
        return c_bar_1, c_bar_2, EI_measure
    elif fairness_criteria == 'pos_pred_rate':
        positive_prob_s_0, positive_prob_s_1 = positive_prediction_ratio(pred_arr, group_arr)
        return positive_prob_s_0, positive_prob_s_1






