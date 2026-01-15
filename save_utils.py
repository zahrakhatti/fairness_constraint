# Enhanced save_utils.py with consistent naming and cleaner logic
import os
import json
import shutil
import pandas as pd
import torch
from datetime import datetime


# ============================================================================
# Helper Functions
# ============================================================================

def format_param_for_path(value):
    """Format parameter value consistently for file paths."""
    if value == float('inf') or value == 'inf':
        return "inf"
    if isinstance(value, float):
        return f"{value:.1e}"
    return str(value)


def get_experiment_id(eps, lmbd):
    """Generate consistent experiment ID from epsilon and lambda."""
    eps_str = format_param_for_path(eps)
    lmbd_str = format_param_for_path(lmbd)
    return f"eps_{eps_str}_lambda_{lmbd_str}"


def get_model_state(model):
    """Extract state dict from model, handling DataParallel wrapper."""
    if hasattr(model, 'module'):
        return model.module.state_dict()
    return model.state_dict()


# ============================================================================
# Results Manager Class
# ============================================================================

class ResultsManager:
    """Manage model checkpoints, results, and experiment logs."""

    def __init__(self, base_path="output"):
        self.base_path = base_path

    def save_results(self, results, parameters, filepath, result_type="train"):
        """
        Save results to CSV with parameters as header comments.

        Args:
            results: List of dictionaries containing results
            parameters: Dictionary of parameters
            filepath: Path to save CSV file
            result_type: Type of results ("train" or "test")
        """
        if not results:
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        backup_path = filepath + '.bak'

        # Create backup if file exists
        if os.path.exists(filepath):
            shutil.copy2(filepath, backup_path)

        try:
            # Save parameters.json
            params_path = os.path.join(os.path.dirname(filepath), "parameters.json")
            with open(params_path, 'w') as f:
                json.dump(parameters, f, indent=2)

            # Create header with parameters
            header_lines = [f"# {result_type.title()} Results", "# Parameters:"]
            header_lines.extend(f"# {key}: {value}" for key, value in parameters.items())
            header_lines.append("#" + "-" * 25 + "\n")
            description = "\n".join(header_lines)

            # Save DataFrame with header
            df = pd.DataFrame(results)
            if 'epoch' in df.columns:
                df = df.sort_values(['epoch'])

            with open(filepath, 'w') as f:
                f.write(description)
                df.to_csv(f, index=False)

            # Remove backup on success
            if os.path.exists(backup_path):
                os.remove(backup_path)

            print(f'{result_type.title()} results saved at {filepath}')

        except Exception as e:
            # Restore from backup on error
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, filepath)
            raise e

    def save_epoch_model(self, model, epoch, base_dir, eps, lmbd, parameters):
        """
        Save model checkpoint for a specific epoch.

        Args:
            model: PyTorch model
            epoch: Current epoch number
            base_dir: Base directory for the model type
            eps: Epsilon value
            lmbd: Lambda value
            parameters: Parameters dictionary
        """
        experiment_id = get_experiment_id(eps, lmbd)
        checkpoint_dir = os.path.join(base_dir, "checkpoints", experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")

        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': get_model_state(model),
                'parameters': parameters,
                'model_architecture': str(model),
                'save_timestamp': datetime.now().isoformat()
            }
            torch.save(checkpoint, checkpoint_path)

            # Print progress every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                print(f'Epoch {epoch} model saved at {checkpoint_path}')

            return checkpoint_path

        except Exception as e:
            print(f"Error saving epoch model: {e}")
            raise e

    def save_final_model(self, model, model_filepath, parameters):
        """
        Save the final trained model.

        Args:
            model: Trained PyTorch model
            model_filepath: Full path to save the model
            parameters: Parameters dictionary
        """
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)

        try:
            checkpoint = {
                'model_state_dict': get_model_state(model),
                'parameters': parameters,
                'model_architecture': str(model),
                'save_timestamp': datetime.now().isoformat()
            }
            torch.save(checkpoint, model_filepath)

            # Save parameters.json in model directory
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
        Load a saved model checkpoint.

        Args:
            model_filepath: Path to the saved model file
            device: Device to load the model on

        Returns:
            Tuple of (loaded_model, parameters)
        """
        try:
            checkpoint = torch.load(model_filepath, map_location=device, weights_only=True)
            parameters = checkpoint['parameters']

            # Import here to avoid circular imports
            from models import Feedforward

            # Initialize model with saved parameters
            model = Feedforward(
                input_size=parameters['input_size'],
                hidden_sizes=parameters['hidden_sizes'],
                num_classes=parameters['num_classes']
            ).to(device)

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            return model, parameters

        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def log_experiment_metrics(self, base_dir, model_type, eps, lmbd, metrics_dict):
        """
        Log experiment metrics to CSV file.

        Args:
            base_dir: Base directory for experiments
            model_type: Type of model
            eps: Epsilon value
            lmbd: Lambda value
            metrics_dict: Dictionary of metrics to log
        """
        try:
            metrics_file = os.path.join(base_dir, f"model_{model_type}_experiments.csv")
            experiment_id = get_experiment_id(eps, lmbd)

            # Prepare row data
            row_data = {
                'experiment_id': experiment_id,
                'model_type': model_type,
                'eps': eps,
                'lambda': lmbd,
                **metrics_dict,
                'completion_timestamp': datetime.now().isoformat()
            }

            # Load existing data or create new
            if os.path.exists(metrics_file):
                df_existing = pd.read_csv(metrics_file)
                mask = (df_existing['experiment_id'] == experiment_id)

                if mask.any():
                    # Update existing row
                    idx = df_existing[mask].index[0]
                    for col, value in row_data.items():
                        df_existing.loc[idx, col] = value
                    df_final = df_existing
                    print(f'Updated experiment: {experiment_id}')
                else:
                    # Add new row
                    df_new = pd.DataFrame([row_data])
                    df_final = pd.concat([df_existing, df_new], ignore_index=True)
                    print(f'Added new experiment: {experiment_id}')
            else:
                # Create new file
                df_final = pd.DataFrame([row_data])
                print(f'Created experiment log: {experiment_id}')

            # Sort and save
            df_final = df_final.sort_values('experiment_id')
            df_final.to_csv(metrics_file, index=False, float_format='%.6f')

        except Exception as e:
            print(f"Error logging experiment metrics: {e}")
            import traceback
            traceback.print_exc()

    def get_experiment_summary(self, base_dir, model_type):
        """Get summary of all experiments for a model type."""
        metrics_file = os.path.join(base_dir, f"model_{model_type}_experiments.csv")
        if os.path.exists(metrics_file):
            return pd.read_csv(metrics_file)
        print(f"No experiment log found at {metrics_file}")
        return pd.DataFrame()

    def create_directory_structure(self, base_dir):
        """
        Create standard directory structure for experiments.

        Returns:
            Tuple of directory paths: (model_dir, train_results_dir, test_results_dir, checkpoints_dir, logs_dir)
        """
        dirs = {
            'model': os.path.join(base_dir, "model"),
            'train_results': os.path.join(base_dir, "train_results"),
            'test_results': os.path.join(base_dir, "test_results"),
            'checkpoints': os.path.join(base_dir, "checkpoints"),
            'logs': os.path.join(base_dir, "logs")
        }

        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        return tuple(dirs.values())

    def create_progress_report(self, output_dir, datasets, model_types):
        """Create comprehensive progress report across all experiments."""
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
                return None

            # Combine and save
            combined_df = pd.concat(all_experiments, ignore_index=True)
            report_path = os.path.join(output_dir, "progress_report.csv")
            combined_df.to_csv(report_path, index=False, float_format='%.6f')

            # Print summary
            if 'total_experiment_time_min' in combined_df.columns:
                total_hours = combined_df['total_experiment_time_min'].sum() / 60
                print(f"Total computation time: {total_hours:.2f} hours")

            print(f"Progress report saved to: {report_path}")
            return combined_df

        except Exception as e:
            print(f"Error creating progress report: {e}")
            return None

    def query_results(self, search_path, result_type="train"):
        """
        Query results from a directory path.

        Args:
            search_path: Directory to search
            result_type: "train" or "test"

        Returns:
            DataFrame with combined results
        """
        matching_results = []

        for root, _, files in os.walk(search_path):
            for file in files:
                if file.endswith('.csv') and not file.endswith('.bak'):
                    try:
                        filepath = os.path.join(root, file)
                        param_path = os.path.join(os.path.dirname(filepath), "parameters.json")

                        # Load parameters if available
                        if os.path.exists(param_path):
                            with open(param_path) as f:
                                parameters = json.load(f)
                        else:
                            parameters = {}

                        # Load results
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
