

import pandas as pd
import os
from config import get_args
import json

args = get_args()

def format_value(val):
    return 0 if val == 0 else val

def get_final_epoch_results(results_path, epsilon):
    try:
        results_df = pd.read_csv(results_path, comment='#', engine='python')
        # Simply return the last row without checking the path
        return results_df.iloc[-1]
    except Exception as e:
        print(f"Error reading {results_path}: {str(e)}")
    return None

def safe_get_value(series, column_name, default_value=0.0):
    """Safely get a value from a pandas series, return default if column doesn't exist."""
    if column_name in series.index:
        return series[column_name]
    else:
        print(f"Warning: Column '{column_name}' not found. Using default value {default_value}")
        return default_value

def get_column_mapping(model_type):
    """Get the correct column names for different model types."""
    # Base mapping that should work for most models
    base_mapping = {
        'acc_overall': 'acc_overall',
        'acc_male': 'acc_male', 
        'acc_female': 'acc_female',
        'const_v_model': 'const_v_model',
        'EI_model': 'EI_model',
        'const_v_measure_di': 'const_v_measure_di',
        'const_v_measure_ei': 'const_v_measure_ei'
    }
    
    # Model-specific overrides if needed
    if model_type == 8:
        base_mapping.update({
        })
    elif model_type == 9:
        base_mapping.update({
        })
    
    return base_mapping

def create_constraint_comparison(dataset, model_type, output_dir, args):
    epsilon_values = ["8.0e-01", "9.0e-01"]
    tables = {}
    
    for eps in epsilon_values: # get each one from its own directory
        scenarios = [
            (f"Only DI (δ={format_value(eps)})", eps, "0.0e+00", 2),  # Added model type
            (f"Only EI (δ={format_value(eps)})", "0.0e+00", eps, 9),   # Added model type  
            (f"Both DI-EI (δ={format_value(eps)})", eps, eps, 8)       # Added model type
        ]
        
        results_data = []
        for desc, eps_di, eps_ei, current_model_type in scenarios:
            if eps_di == "0.0e+00":
                # Use args.data_plot instead of hardcoded "results"
                if args.scaled == False:
                    base_dir = f"output/scaled_{args.scaled}/{dataset}/full_batch_True/9/train_results"
                else:
                    base_dir = f"output/{dataset}/full_batch_True/9/train_results"
                eps_str = f"eps_{eps_ei}_lambda_inf.csv"
                actual_model_type = 9
            elif eps_ei == "0.0e+00":
                # Use args.data_plot instead of hardcoded "results"
                if args.scaled == False:
                    base_dir = f"output/scaled_{args.scaled}/{dataset}/full_batch_True/2/train_results"
                else:
                    base_dir = f"output/{dataset}/full_batch_True/2/train_results"
                eps_str = f"eps_{eps_di}_lambda_inf.csv"
                actual_model_type = 2
            elif eps_di == eps_ei:
                # Use args.data_plot instead of hardcoded "results"
                if args.scaled == False:
                    base_dir = f"output/scaled_{args.scaled}/{dataset}/full_batch_True/8/train_results"
                else:
                    base_dir = f"output/{dataset}/full_batch_True/8/train_results"
                eps_str = f"eps_{eps_di}_lambda_inf.csv"
                actual_model_type = 8
            
            results_path = os.path.join(base_dir, eps_str)
            
            # Only proceed if the file exists
            if not os.path.exists(results_path):
                print(f"Warning: File not found: {results_path}")
                continue
                
            final_results = get_final_epoch_results(results_path, float(eps_di) if eps_di != "0.0e+00" else float(eps_ei))
            if final_results is not None:
                print(f"Processing {results_path} (Model Type {actual_model_type})")
                print(f"Available columns: {final_results.index.tolist()}")
                
                # Get the correct column mapping for this model type
                column_mapping = get_column_mapping(actual_model_type)
                
                # Use safe_get_value with the mapped column names
                results_data.append({
                    'Scenario': desc,
                    'Overall Acc': safe_get_value(final_results, column_mapping['acc_overall']),
                    'Male Acc': safe_get_value(final_results, column_mapping['acc_male']),
                    'Female Acc': safe_get_value(final_results, column_mapping['acc_female']),
                    'DI constraint Violation Model': safe_get_value(final_results, column_mapping['const_v_model']),
                    'EI constraint Violation Model': safe_get_value(final_results, column_mapping['EI_model']),
                    'DI Constraint Violation Measure': safe_get_value(final_results, column_mapping['const_v_measure_di']),
                    'EI Constraint Violation Measure': safe_get_value(final_results, column_mapping['const_v_measure_ei']),
                })
        
        if results_data:
            df = pd.DataFrame(results_data)
            display_df = df.round(4)
            csv_path = os.path.join(output_dir, f"constraint_comparison_eps_{eps}.csv")
            display_df.to_csv(csv_path, index=False)
            
            print(f"\nResults for δ = {eps}:")
            print("=" * 100)
            print(display_df.to_string(index=False))
            print(f"\nResults saved to: {csv_path}")
            
            tables[eps] = df
    
    return tables