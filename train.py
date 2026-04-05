"""
Main training script for fine-tuning language models.
"""

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
from src.utils.config_loader import load_config, validate_config
from src.utils.logger import setup_logger
from src.training.trainer import Trainer

logger = setup_logger("main")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune language models with HuggingFace Transformers"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Store the original config file path for record keeping
    config["_config_file_path"] = str(Path(args.config).resolve())
    
    # Override config with command line argument if provided
    if args.resume_from_checkpoint:
        config["resume_from_checkpoint"] = args.resume_from_checkpoint
        logger.info(f"Will resume training from checkpoint: {args.resume_from_checkpoint}")
    
    # Validate required keys
    required_keys = [
        "model_name",
        "output_dir",
        "training_mode"
    ]
    validate_config(config, required_keys)
    
    # Check dataset configuration: must have either dataset_path or batch_dirs
    if "dataset_path" not in config and "batch_dirs" not in config:
        raise ValueError("Config must contain either 'dataset_path' or 'batch_dirs'")
    
    # Create trainer and start training
    logger.info("Initializing trainer")
    trainer = Trainer(config)
    
    logger.info("Starting training process")
    trainer.train()
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()

