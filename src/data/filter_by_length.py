"""
Filter tokenized dataset by sequence length with batch processing support.
"""

import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("filter_by_length")


def load_dataset_from_dir(input_dir: str) -> Dataset:
    """
    Load dataset from directory. Supports both simple and batch mode datasets.
    
    Args:
        input_dir: Directory containing the dataset
    
    Returns:
        Concatenated dataset
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Check if it's a batch mode dataset (contains batch_XXXX subdirectories)
    batch_dirs = sorted(list(input_path.glob("batch_*")))
    
    if batch_dirs:
        logger.info(f"Detected batch mode dataset with {len(batch_dirs)} batches")
        logger.info("Loading and concatenating all batches...")
        
        datasets = []
        for batch_dir in tqdm(batch_dirs, desc="Loading batches"):
            try:
                batch_dataset = Dataset.load_from_disk(str(batch_dir))
                datasets.append(batch_dataset)
                logger.info(f"Loaded {batch_dir.name}: {len(batch_dataset)} samples")
            except Exception as e:
                logger.error(f"Failed to load {batch_dir}: {e}")
        
        if not datasets:
            raise ValueError(f"No valid batches found in {input_dir}")
        
        # Concatenate all batches
        full_dataset = concatenate_datasets(datasets)
        logger.info(f"Total samples loaded: {len(full_dataset)}")
        return full_dataset
    
    else:
        # Try to load as simple mode dataset (DatasetDict)
        logger.info("Detected simple mode dataset")
        try:
            dataset_dict = DatasetDict.load_from_disk(input_dir)
            
            # Concatenate train and test if both exist
            datasets = []
            if 'train' in dataset_dict:
                datasets.append(dataset_dict['train'])
                logger.info(f"Loaded train set: {len(dataset_dict['train'])} samples")
            if 'test' in dataset_dict:
                datasets.append(dataset_dict['test'])
                logger.info(f"Loaded test set: {len(dataset_dict['test'])} samples")
            
            if not datasets:
                raise ValueError(f"No train or test set found in {input_dir}")
            
            full_dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
            logger.info(f"Total samples loaded: {len(full_dataset)}")
            return full_dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise


def filter_by_length(
    dataset: Dataset,
    min_length: int = 0,
    max_length: int = float('inf'),
    num_proc: Optional[int] = None
) -> Dataset:
    """
    Filter dataset by sequence length.
    
    Args:
        dataset: Input dataset with 'input_ids' field
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)
        num_proc: Number of processes for parallel processing
    
    Returns:
        Filtered dataset
    """
    logger.info(f"Filtering sequences with length in range [{min_length}, {max_length}]")
    if num_proc:
        logger.info(f"Using {num_proc} processes for parallel filtering")
    
    def length_filter(example):
        seq_length = len(example['input_ids'])
        return min_length <= seq_length <= max_length
    
    filtered_dataset = dataset.filter(
        length_filter,
        num_proc=num_proc,
        desc="Filtering by length"
    )
    
    logger.info(f"Original samples: {len(dataset)}")
    logger.info(f"Filtered samples: {len(filtered_dataset)}")
    logger.info(f"Removed samples: {len(dataset) - len(filtered_dataset)}")
    
    return filtered_dataset


def filter_dataset_simple(
    input_dir: str,
    output_dir: str,
    min_length: int = 0,
    max_length: int = float('inf'),
    seed: int = 42,
    num_proc: Optional[int] = None
) -> None:
    """
    Filter dataset by sequence length (simple mode - one-time processing).
    
    This function loads all data at once and processes it in a single pass.
    Suitable for small to medium datasets or systems with sufficient memory.
    
    Args:
        input_dir: Directory containing the input dataset
        output_dir: Directory to save the filtered dataset
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)
        seed: Random seed (kept for compatibility)
        num_proc: Number of processes for parallel processing
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset_from_dir(input_dir)
    
    # Filter by length
    logger.info("Filtering dataset by sequence length...")
    filtered_dataset = filter_by_length(dataset, min_length, max_length, num_proc)
    
    if len(filtered_dataset) == 0:
        logger.error("No samples remaining after filtering. Please adjust length thresholds.")
        return
    
    # Save as simple dataset (no train/test split, just save as-is)
    logger.info(f"Saving filtered dataset to {output_dir}")
    filtered_dataset.save_to_disk(output_dir)
    
    logger.info("Dataset filtering completed")
    logger.info(f"Filtered samples: {len(filtered_dataset)}")


def filter_dataset_batch(
    input_dir: str,
    output_dir: str,
    min_length: int = 0,
    max_length: int = float('inf'),
    seed: int = 42,
    num_proc: Optional[int] = None,
    batch_size: int = 5000
) -> None:
    """
    Filter dataset by sequence length (batch mode - memory efficient).
    
    This function processes large datasets in batches to avoid memory issues.
    Each batch is saved as an independent dataset that can be used directly by
    the training code.
    
    Args:
        input_dir: Directory containing the input dataset
        output_dir: Directory to save the filtered batches (each batch saved as batch_XXXX)
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)
        seed: Random seed (kept for compatibility)
        num_proc: Number of processes for parallel processing
        batch_size: Number of samples per batch (default: 5000)
    """
    import gc
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset_from_dir(input_dir)
    
    # Filter by length
    logger.info("Filtering dataset by sequence length...")
    filtered_dataset = filter_by_length(dataset, min_length, max_length, num_proc)
    
    if len(filtered_dataset) == 0:
        logger.error("No samples remaining after filtering. Please adjust length thresholds.")
        return
    
    total_samples = len(filtered_dataset)
    logger.info(f"[BATCH MODE] Saving in batches of {batch_size} samples")
    logger.info(f"[BATCH MODE] Each batch will be saved independently as batch_XXXX")
    
    # Process in batches and save each batch independently
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        
        logger.info(f"Processing batch {batch_idx}/{num_batches - 1}: samples {start_idx} to {end_idx - 1}")
        
        # Select samples for this batch
        batch_dataset = filtered_dataset.select(range(start_idx, end_idx))
        
        # Save this batch directly to output directory
        batch_dir = output_path / f"batch_{batch_idx:04d}"
        batch_dataset.save_to_disk(str(batch_dir))
        
        batch_samples = len(batch_dataset)
        logger.info(f"Batch {batch_idx} completed: {batch_samples} samples, saved to {batch_dir}")
        
        # Clear memory
        del batch_dataset
        gc.collect()
    
    logger.info("Dataset filtering completed")
    logger.info(f"Total batches created: {num_batches}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Batches saved to: {output_dir}/batch_XXXX")


def filter_dataset(
    input_dir: str,
    output_dir: str,
    min_length: int = 0,
    max_length: int = float('inf'),
    seed: int = 42,
    num_proc: Optional[int] = None,
    batch_mode: bool = False,
    batch_size: int = 5000
) -> None:
    """
    Filter dataset by sequence length.
    
    Args:
        input_dir: Directory containing the input dataset
        output_dir: Directory to save the filtered dataset
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)
        seed: Random seed
        num_proc: Number of processes for parallel processing
        batch_mode: If True, use batch processing mode (memory efficient for large datasets)
                    If False, use simple mode (faster, but requires more memory)
        batch_size: Number of samples to save per batch (only used if batch_mode=True)
    """
    if batch_mode:
        logger.info("Using BATCH MODE (memory efficient for large datasets)")
        filter_dataset_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            min_length=min_length,
            max_length=max_length,
            seed=seed,
            num_proc=num_proc,
            batch_size=batch_size
        )
    else:
        logger.info("Using SIMPLE MODE (one-time processing)")
        filter_dataset_simple(
            input_dir=input_dir,
            output_dir=output_dir,
            min_length=min_length,
            max_length=max_length,
            seed=seed,
            num_proc=num_proc
        )


def main():
    parser = argparse.ArgumentParser(
        description="Filter tokenized dataset by sequence length"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the input dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save filtered dataset"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=0,
        help="Minimum sequence length (inclusive, default: 0)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=float('inf'),
        help="Maximum sequence length (inclusive, default: inf)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for parallel processing"
    )
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Enable batch processing mode for large datasets (memory efficient)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Number of samples to save per batch (only used with --batch_mode, default: 5000)"
    )
    
    args = parser.parse_args()
    
    filter_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
        num_proc=args.num_proc,
        batch_mode=args.batch_mode,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

