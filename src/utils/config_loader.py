"""
Configuration loader for YAML config files.
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Dictionary containing configuration parameters
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any], required_keys: list) -> None:
    """
    Validate that all required keys exist in the configuration.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key names
    
    Raises:
        ValueError: If any required key is missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

