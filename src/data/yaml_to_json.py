"""
Convert YAML files to JSON format for better compatibility with Qwen models.
Supports float truncation to reduce file size and token count.
"""

import json
import yaml
import argparse
import re
from pathlib import Path
from typing import Any, Dict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("yaml_to_json")


def rename_object_keys(data: Dict) -> Dict:
    """
    重命名顶层的物体key，将形如 "数字_XXFactory" 的key替换成 "object_1", "object_2" 等
    保持原有顺序
    
    Args:
        data: 字典数据
    
    Returns:
        重命名后的字典
    """
    if not isinstance(data, dict):
        return data
    
    # 匹配形如 "数字_XXFactory" 的模式
    pattern = re.compile(r'^\d+_\w+Factory$')
    
    renamed_data = {}
    object_counter = 1
    
    for key, value in data.items():
        if pattern.match(key):
            new_key = f"object_{object_counter}"
            renamed_data[new_key] = value
            object_counter += 1
        else:
            renamed_data[key] = value
    
    return renamed_data


def truncate_floats(obj: Any, precision: int) -> Any:
    """
    Recursively truncate floating point numbers to specified precision.
    
    Args:
        obj: Object to process (can be dict, list, float, or other types)
        precision: Number of decimal places to keep
    
    Returns:
        Object with truncated floats
    """
    if isinstance(obj, dict):
        return {key: truncate_floats(value, precision) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [truncate_floats(item, precision) for item in obj]
    elif isinstance(obj, float):
        # Round to specified precision
        return round(obj, precision)
    else:
        # Return other types as-is (int, str, bool, None, etc.)
        return obj


def convert_yaml_file(yaml_path: Path, output_dir: Path, precision: int = None, rename_objects: bool = False) -> bool:
    """
    Convert a single YAML file to JSON format with optional float truncation.
    
    Args:
        yaml_path: Path to the YAML file
        output_dir: Directory to save the JSON file
        precision: Number of decimal places to keep for floats (None = no truncation)
        rename_objects: Whether to rename object keys (e.g., "696816_BedFactory" -> "object_1")
    
    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 重命名物体key（如果启用）
        if rename_objects:
            data = rename_object_keys(data)
        
        # Truncate floats if precision is specified
        if precision is not None:
            data = truncate_floats(data, precision)
        
        # Create output path with .json extension
        json_path = output_dir / f"{yaml_path.stem}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Failed to convert {yaml_path}: {e}")
        return False


def convert_yaml_to_json(
    input_dir: str,
    output_dir: str,
    num_workers: int = 4,
    precision: int = None,
    rename_objects: bool = False
) -> None:
    """
    Convert all YAML files in a directory to JSON format with optional float truncation.
    
    Args:
        input_dir: Directory containing YAML files
        output_dir: Directory to save JSON files
        num_workers: Number of parallel workers for processing
        precision: Number of decimal places to keep for floats (None = no truncation)
        rename_objects: Whether to rename object keys (e.g., "696816_BedFactory" -> "object_1")
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all YAML files
    yaml_files = list(input_path.glob("*.yaml")) + list(input_path.glob("*.yml"))
    
    if not yaml_files:
        logger.warning(f"No YAML files found in {input_dir}")
        return
    
    logger.info(f"Found {len(yaml_files)} YAML files to convert")
    if precision is not None:
        logger.info(f"Float precision truncation enabled: {precision} decimal places")
    if rename_objects:
        logger.info(f"Object key renaming enabled")
    
    # Convert files in parallel
    convert_fn = partial(convert_yaml_file, output_dir=output_path, precision=precision, rename_objects=rename_objects)
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_fn, yaml_files),
            total=len(yaml_files),
            desc="Converting YAML to JSON"
        ))
    
    successful = sum(results)
    logger.info(f"Successfully converted {successful}/{len(yaml_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YAML files to JSON format with optional float truncation and object key renaming"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing YAML files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save JSON files"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=None,
        help="Number of decimal places to keep for floats (default: None, no truncation). "
             "Example: --precision 4 converts 0.01983328167140487 to 0.0198"
    )
    parser.add_argument(
        "--rename_objects",
        action="store_true",
        help="Rename object keys from '数字_XXFactory' pattern to 'object_1', 'object_2', etc."
    )
    
    args = parser.parse_args()
    
    convert_yaml_to_json(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        precision=args.precision,
        rename_objects=args.rename_objects
    )


if __name__ == "__main__":
    main()

