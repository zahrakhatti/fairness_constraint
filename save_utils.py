# Enhanced save_utils.py with consistent naming
import os
import json
import shutil
import pandas as pd
import torch
import time
from datetime import datetime

class ResultsManager:
    def __init__(self, base_path="output"):
       self.base_path = base_path

    def save_results(self, results, parameters, filepath, result_type="train"):
        if not results:
            return
        
        # Use the filepath directly
        new_filepath = filepath
        
        os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
        backup_path = new_filepath + '.bak'
        
        if os.path.exists(new_filepath):
            shutil.copy2(new_filepath, backup_path)
        
        try:
            # Save parameters.json
            params_path = os.path.join(os.path.dirname(new_filepath), "parameters.json")
            with open(params_path, 'w') as f:
                json.dump(parameters, f, indent=2)
            
            # Create description
            description = f"# {result_type.title()} Results\n"
            description += "# Parameters:\n"
            for key, value in parameters.items():
                description += f"# {key}: {value}\n"
            description += "#" + "-"*25 + "\n"
            
            # Save DataFrame with description
            df = pd.DataFrame(results)
            if 'epoch' in df.columns:
                df = df.sort_values(['epoch'])
            
            with open(new_filepath, 'w') as f:
                f.write(description)
                df.to_csv(f, index=False)
            
            if os.path.exists(backup_path):
                os.remove(backup_path)
                
            print(f'{result_type.title()} results saved at {new_filepath}')
                
        except Exception as e:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, new_filepath)
            raise e

    def save_epoch_model(self, model, epoch, base_dir, eps, lmbd, parameters):
        """
        Save model for a specific epoch in a subdirectory.
        FIXED: Use consistent naming format that matches testing expectations.
        
        Args:
            model: The PyTorch model
            epoch: Current epoch number
            base_dir: Base directory for the model type
            eps: Epsilon value
            lmbd: Lambda value
            parameters: Parameters dictionary
        """
        # Use consistent format (no scientific notation for checkpoint dirs)
        eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
        lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
        
        # Create subdirectory in checkpoints
        checkpoint_subdir = os.path.join(base_dir, "checkpoints", f"eps_{eps_str}_lambda_{lmbd_str}")
        os.makedirs(checkpoint_subdir, exist_ok=True)
        
        # Save model for this epoch
        epoch_filename = f"epoch_{epoch}.pth"
        epoch_filepath = os.path.join(checkpoint_subdir, epoch_filename)
        
        try:
            # Get model state dict
            if hasattr(model, 'module'):  # Handle DataParallel models
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'parameters': parameters,
                'model_architecture': str(model),
                'save_timestamp': datetime.now().isoformat()
            }, epoch_filepath)
            
            # Print every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                print(f'Epoch {epoch} model saved at {epoch_filepath}')
            return epoch_filepath
            
        except Exception as e:
            print(f"Error saving epoch model: {e}")
            raise e

    def save_final_model(self, model, model_filepath, parameters):
        """
        Save the final trained model in the model directory.
        
        Args:
            model: The trained PyTorch model
            model_filepath: Full path where to save the model
            parameters: Parameters dictionary for reference
        """
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
        
        try:
            # Save model state dict
            if hasattr(model, 'module'):  # Handle DataParallel models
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            
            # Save model with parameters for reference
            torch.save({
                'model_state_dict': model_state,
                'parameters': parameters,
                'model_architecture': str(model),
                'save_timestamp': datetime.now().isoformat()
            }, model_filepath)
            
            # Also save parameters.json in model directory
            params_path = os.path.join(os.path.dirname(model_filepath), "parameters.json")
            with open(params_path, 'w') as f:
                json.dump(parameters, f, indent=2)
            
            print(f'Final model saved at {model_filepath}')
            return model_filepath
            
        except Exception as e:
            print(f"Error saving final model: {e}")
            raise e

    def load_model(self, model_filepath, device='cpu'):
        """
        Load a saved model and return both the model and its parameters.
        
        Args:
            model_filepath: Path to the saved model file
            device: Device to load the model on
            
        Returns:
            Tuple of (loaded_model, parameters)
        """
        try:
            checkpoint = torch.load(model_filepath, map_location=device, weights_only= True)
            parameters = checkpoint['parameters']
            
            # Import here to avoid circular imports
            from models import Feedforward
            
            # Initialize model with parameters from checkpoint
            model = Feedforward(
                input_size=parameters['input_size'],
                hidden_sizes=parameters['hidden_sizes'], 
                num_classes=parameters['num_classes']
            ).to(device)
            
            # Load the state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            
            return model, parameters
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def log_experiment_metrics(self, base_dir, model_type, eps, lmbd, metrics_dict):
        """Log experiment metrics to a CSV file for tracking resource usage and performance."""
        try:
            metrics_file = os.path.join(base_dir, f"model_{model_type}_experiments.csv")
            
            eps_str = f"{eps:.1e}" if eps != float('inf') else "inf"
            lmbd_str = f"{lmbd}" if lmbd != float('inf') else "inf"
            experiment_id = f"eps_{eps_str}_lambda_{lmbd_str}"
            
            row_data = {
                'experiment_id': experiment_id,
                'model_type': model_type,
                'eps': eps,
                'lambda': lmbd,
                **metrics_dict
            }
            
            file_exists = os.path.exists(metrics_file)
            
            if file_exists:
                df_existing = pd.read_csv(metrics_file)
                mask = (df_existing['experiment_id'] == experiment_id)
                
                if mask.any():
                    existing_row_idx = df_existing[mask].index[0]
                    
                    for col, value in row_data.items():
                        df_existing.loc[existing_row_idx, col] = value
                    
                    if 'completion_timestamp' not in df_existing.columns or pd.isna(df_existing.loc[existing_row_idx, 'completion_timestamp']):
                        df_existing.loc[existing_row_idx, 'completion_timestamp'] = datetime.now().isoformat()
                    
                    df_final = df_existing
                    print(f'Updated existing experiment: {experiment_id} (preserved existing data)')
                else:
                    # Add new row
                    row_data['completion_timestamp'] = datetime.now().isoformat()
                    df_new = pd.DataFrame([row_data])
                    df_final = pd.concat([df_existing, df_new], ignore_index=True)
                    print(f'Added new experiment: {experiment_id}')
            else:
                # Create new file
                row_data['completion_timestamp'] = datetime.now().isoformat()
                df_final = pd.DataFrame([row_data])
                print(f'Created new experiment log: {experiment_id}')
            
            # Sort by experiment_id for consistency
            df_final = df_final.sort_values('experiment_id')
            
            # Save with proper formatting
            df_final.to_csv(metrics_file, index=False, float_format='%.6f')
            

        except Exception as e:
            print(f"Error logging experiment metrics: {e}")
            import traceback
            traceback.print_exc()

    def get_experiment_summary(self, base_dir, model_type):
        """Get summary of all experiments for a model type."""
        try:
            metrics_file = os.path.join(base_dir, f"model_{model_type}_experiments.csv")
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                return df
            else:
                print(f"No experiment log found at {metrics_file}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading experiment summary: {e}")
            return pd.DataFrame()

    def create_directory_structure(self, base_dir):
        """
        Create the directory structure: model, train_results, test_results, checkpoints, logs
        
        Args:
            base_dir: Base directory path
            
        Returns:
            Paths to the directories
        """
        model_dir = os.path.join(base_dir, "model")
        train_results_dir = os.path.join(base_dir, "train_results")
        test_results_dir = os.path.join(base_dir, "test_results")
        checkpoints_dir = os.path.join(base_dir, "checkpoints")
        logs_dir = os.path.join(base_dir, "logs")
        
        for dir_path in [model_dir, train_results_dir, test_results_dir, checkpoints_dir, logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        return model_dir, train_results_dir, test_results_dir, checkpoints_dir, logs_dir

    def create_progress_report(self, output_dir, datasets, model_types):
        """Create a comprehensive progress report across all experiments."""
        try:
            all_experiments = []
            
            for dataset in datasets:
                for model_type in model_types:
                    base_dir = os.path.join(output_dir, dataset, f"full_batch_True", str(model_type))
                    experiment_df = self.get_experiment_summary(base_dir, model_type)
                    
                    if not experiment_df.empty:
                        experiment_df['dataset'] = dataset
                        experiment_df['model_type'] = model_type
                        all_experiments.append(experiment_df)
            
            if not all_experiments:
                print("No experiments found for progress report")
                return
            
            # Combine all experiments
            combined_df = pd.concat(all_experiments, ignore_index=True)
            
            # Create progress report
            report_path = os.path.join(output_dir, "progress_report.csv")
            combined_df.to_csv(report_path, index=False, float_format='%.6f')
            
            if 'total_experiment_time_min' in combined_df.columns:
                total_time_hours = combined_df['total_experiment_time_min'].sum() / 60
                print(f"Total computation time: {total_time_hours:.2f} hours")
            
            print(f"Progress report saved to: {report_path}")
            
            return combined_df
            
        except Exception as e:
            print(f"Error creating progress report: {e}")
            return None

    def query_results(self, search_path, result_type="train"):
        """
        Query results from a specific directory path.
        
        Args:
            search_path: Directory path to search
            result_type: "train" or "test" to specify which results to query
        """
        matching_results = []
        
        for root, _, files in os.walk(search_path):
            for file in files:
                if file.endswith('.csv') and not file.endswith('.bak'):
                    try:
                        filepath = os.path.join(root, file)
                        param_path = os.path.join(os.path.dirname(filepath), "parameters.json")
                        
                        if os.path.exists(param_path):
                            with open(param_path) as f:
                                parameters = json.load(f)
                        else:
                            parameters = {}
                            
                        results_df = pd.read_csv(filepath, comment='#')
                        results_df['source_file'] = filepath
                        results_df['parameters'] = str(parameters)
                        results_df['result_type'] = result_type
                        matching_results.append(results_df)
                            
                    except Exception as e:
                        print(f"Error reading {filepath}: {str(e)}")
        
        if not matching_results:
            return pd.DataFrame()
            
        return pd.concat(matching_results, ignore_index=True)

    def get_parameter_summary(self, search_path):
        """Get parameter summary from a specific directory."""
        all_params = {}
        
        # Check train_results, test_results, and model directories
        for result_type in ["train_results", "test_results", "model"]:
            type_search_path = os.path.join(search_path, result_type)
            
            for root, _, files in os.walk(type_search_path):
                for file in files:
                    if file == 'parameters.json':
                        try:
                            with open(os.path.join(root, file)) as f:
                                params = json.load(f)
                                for k, v in params.items():
                                    if k not in all_params:
                                        all_params[k] = set()
                                    all_params[k].add(v)
                        except:
                            continue

        print("\nParameters Summary:")
        for param, values in all_params.items():
            values = sorted(list(values))
            print(f"\n{param}:")
            if len(values) > 10:
                print(f"  Range: {min(values)} to {max(values)}")
                print(f"  Count: {len(values)}")
            else:
                print(f"  Values: {values}")

    def fix_checkpoint_naming_mismatch(self, base_dir, eps, lmbd):
        """
        Helper function to fix naming mismatches between checkpoint directories.
        This can be used to rename existing checkpoint directories to the consistent format.
        """
        # Current scientific notation format
        eps_sci = f"{eps:.1e}" if eps != float('inf') else "inf"
        lmbd_sci = f"{lmbd}" if lmbd != float('inf') else "inf"
        sci_dir = os.path.join(base_dir, "checkpoints", f"eps_{eps_sci}_lambda_{lmbd_sci}")
        
        # Simple format that should be used
        eps_simple = f"{eps:.1e}" if eps != float('inf') else "inf"
        lmbd_simple = f"{lmbd}" if lmbd != float('inf') else "inf"
        simple_dir = os.path.join(base_dir, "checkpoints", f"eps_{eps_simple}_lambda_{lmbd_simple}")
        
        if os.path.exists(sci_dir) and not os.path.exists(simple_dir):
            try:
                os.rename(sci_dir, simple_dir)
                print(f"Renamed checkpoint directory: {sci_dir} -> {simple_dir}")
                return True
            except Exception as e:
                print(f"Error renaming checkpoint directory: {e}")
                return False
        
        return False