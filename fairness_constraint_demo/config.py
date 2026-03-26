import json
import os

# Dataset-specific configurations 
DATASET_CONFIGS = {
    "adult": {
        "input_size": 90,
        "hidden_sizes": [128, 64],
        "lr": 0.5,
        "sf": "sex"
    },
    "law": {
        "input_size": 11,
        "hidden_sizes": [],
        "lr": 0.5,
        "sf": "race"
    },
    "compas": {
        "input_size": 4,
        "hidden_sizes": [16, 8],
        "lr": 0.01,
        "sf": "race"
    },
    "acsincome": {
        "input_size": 10,
        "hidden_sizes": [256],
        "lr": 0.5,
        "sf": "sex"
    },
    "acsemployment": {
        "input_size": 17,
        "hidden_sizes": [64, 32],
        "lr": 0.005,
        "sf": "sex"
    },
    "dutch": {
        "input_size": 11,
        "hidden_sizes": [32],
        "lr": 0.5,
        "sf": "sex"
    },
}

# Global configuration object
_config = None

def load_config():
    """Load configuration from file, apply dataset-specific settings."""
    global _config
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use relative path for config.json
    file_path = os.path.join(current_dir, 'config.json')
    
    try:
        with open(file_path, 'r') as f:
            _config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find config.json in {current_dir}")
    
    # Apply dataset-specific settings
    dataset = _config.get("dataset")
    if dataset in DATASET_CONFIGS:
        _config.update(DATASET_CONFIGS[dataset])

def get_args():
    """Get configuration as an object with attributes."""
    global _config
    
    # Load config if not already loaded
    if _config is None:
        load_config()
    
    # Convert the dictionary to an object with attributes
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
        
        def update(self, new_values):
            """Update configuration values."""
            self.__dict__.update(new_values)
            global _config
            _config.update(new_values)
    
    return Config(**_config)

def update_config(new_values):
    """Update configuration values globally."""
    global _config
    
    # Load config if not already loaded
    if _config is None:
        load_config()
    
    _config.update(new_values)
    
    # If dataset changed, apply dataset-specific settings
    if "dataset" in new_values and new_values["dataset"] in DATASET_CONFIGS:
        _config.update(DATASET_CONFIGS[new_values["dataset"]])

def get_config_value(key, default=None):
    """Get a specific configuration value."""
    global _config
    
    # Load config if not already loaded
    if _config is None:
        load_config()
    
    return _config.get(key, default)