"""
Prepare dataset for fine-tuning with tokenization and HuggingFace Dataset packaging.
Supports both text-only and vision-language (with panorama images) modes.
"""

import json
import argparse
import gc
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor
from datasets import Dataset, DatasetDict, concatenate_datasets

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger
from src.data.loaders import (
    load_json_files,
    load_caption_json_pairs,
    load_panorama_json_pairs,
    load_multiview_json_pairs,
    load_multiview2_json_pairs,
    load_multiview3_json_pairs,
)
from src.data.converters import convert_to_qwen_format
from src.data.tokenization import (
    compute_image_token_count,
    compute_image_token_counts_batch,
    tokenize_function,
    tokenize_vl_function,
    tokenize_multiview_function,
)
from src.data.packing import pack_sequences
from src.data.image_archives import (
    read_archive_info,
    write_archive_info,
    ensure_archives_root,
    create_image_archives,
    create_image_archives_dynamic,
)

logger = setup_logger("prepare_dataset")


# =============================================================================
# Helper Functions
# =============================================================================

def parse_resolution(resolution_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse resolution string like '1024x512' to (width, height) tuple."""
    if not resolution_str:
        return None
    try:
        width, height = map(int, resolution_str.lower().split('x'))
        return (width, height)
    except ValueError:
        logger.warning(f"Invalid resolution format: {resolution_str}. Expected 'WIDTHxHEIGHT'.")
        return None


def load_tokenizer_or_processor(model_name: str, is_vl_mode: bool):
    """Load tokenizer or processor based on mode."""
    if is_vl_mode:
        logger.info(f"Loading VL processor from {model_name}")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = processor.tokenizer
        return processor, tokenizer
    else:
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return None, tokenizer


def load_data_by_mode(
    mode: str,
    json_dir: str,
    caption_dir: Optional[str],
    image_dir: Optional[str],
    num_views_range: Optional[tuple]
) -> List[Dict[str, Any]]:
    """Load data based on mode."""
    if mode == "caption":
        if not caption_dir:
            raise ValueError("caption_dir is required for 'caption' mode")
        logger.info("Loading caption-JSON pairs...")
        return load_caption_json_pairs(json_dir, caption_dir)
    elif mode == "panorama":
        if not image_dir:
            raise ValueError("image_dir is required for 'panorama' mode")
        logger.info("Loading panorama-JSON pairs...")
        return load_panorama_json_pairs(json_dir, image_dir)
    elif mode == "multi_view":
        if not image_dir:
            raise ValueError("image_dir is required for 'multi_view' mode")
        views_range = num_views_range or (4, 8)
        logger.info("Loading multi-view-JSON pairs (orbit layout)...")
        return load_multiview_json_pairs(json_dir, image_dir, views_range)
    elif mode == "multi_view2":
        if not image_dir:
            raise ValueError("image_dir is required for 'multi_view2' mode")
        views_range = num_views_range or (4, 8)
        logger.info("Loading multi-view-JSON pairs (flat layout)...")
        return load_multiview2_json_pairs(json_dir, image_dir, views_range)
    elif mode == "multi_view3":
        if not image_dir:
            raise ValueError("image_dir is required for 'multi_view3' mode")
        logger.info("Loading single front-view-JSON pairs...")
        return load_multiview3_json_pairs(json_dir, image_dir)
    else:
        logger.info("Loading all JSON files...")
        return load_json_files(json_dir)


def create_dataset_from_formatted_data(
    formatted_data: List[Dict[str, Any]],
    mode: str,
    image_archives_root: Optional[Path],
    archive_base_key: Optional[str],
    target_size: Optional[Tuple[int, int]],
    num_proc: int,
    shard_size: int,
    processor=None,
    multiview_min_size: Optional[Tuple[int, int]] = None,
    multiview_max_size: Optional[Tuple[int, int]] = None,
    image_format: str = "png",
    image_quality: int = 85,
) -> Dataset:
    """Create HuggingFace Dataset from formatted data."""
    
    if mode == "panorama":
        logger.info("Creating image archives for panorama data...")
        messages = [item["messages"] for item in formatted_data]
        image_paths = [item.get("image_path") for item in formatted_data]
        
        image_refs = create_image_archives(
            image_paths=image_paths,
            archives_root=image_archives_root,
            target_size=target_size,
            num_proc=num_proc,
            shard_size=shard_size,
            image_format=image_format,
            image_quality=image_quality,
        )
        
        valid_messages, valid_refs = [], []
        for msg, ref in zip(messages, image_refs):
            if ref is not None:
                valid_messages.append(msg)
                valid_refs.append(ref)
        
        if not valid_messages:
            raise ValueError("No valid panorama samples after archiving.")
        
        return Dataset.from_dict({
            "messages": valid_messages,
            "image_ref": valid_refs,
            "image_base": [archive_base_key] * len(valid_refs)
        })
    
    elif mode in ("multi_view", "multi_view2", "multi_view3"):
        logger.info("Creating image archives for multi-view data...")
        messages = [item["messages"] for item in formatted_data]
        all_image_paths = [item.get("image_paths", []) for item in formatted_data]
        
        # Flatten all image paths
        flat_image_paths = []
        image_counts = []
        for paths in all_image_paths:
            flat_image_paths.extend(paths)
            image_counts.append(len(paths))
        
        # Check if using dynamic resolution mode
        use_dynamic_resolution = multiview_min_size is not None and multiview_max_size is not None
        flat_image_resolutions = None
        
        if use_dynamic_resolution:
            logger.info(f"[动态分辨率模式] 使用分辨率范围 {multiview_min_size} ~ {multiview_max_size}")
            flat_image_refs, flat_image_resolutions = create_image_archives_dynamic(
                image_paths=flat_image_paths,
                archives_root=image_archives_root,
                min_size=multiview_min_size,
                max_size=multiview_max_size,
                num_proc=num_proc,
                shard_size=shard_size,
                image_format=image_format,
                image_quality=image_quality,
            )
        else:
            flat_image_refs = create_image_archives(
                image_paths=flat_image_paths,
                archives_root=image_archives_root,
                target_size=target_size,
                num_proc=num_proc,
                shard_size=shard_size,
                image_format=image_format,
                image_quality=image_quality,
            )
        
        # Compute token counts
        flat_image_token_counts = None
        if use_dynamic_resolution and processor is not None:
            # For dynamic resolution, compute token counts based on actual resolutions
            # 使用缓存避免重复计算相同分辨率的 token count
            resolution_token_cache = {}
            logger.info("[动态分辨率模式] 根据实际采样分辨率计算 image token 数量...")
            unique_resolutions = set(r for r in flat_image_resolutions if r is not None)
            logger.info(f"[动态分辨率模式] 发现 {len(unique_resolutions)} 种不同的分辨率")
            flat_image_token_counts = []
            for resolution in tqdm(flat_image_resolutions, desc="Computing token counts (dynamic)"):
                if resolution is not None:
                    if resolution not in resolution_token_cache:
                        resolution_token_cache[resolution] = compute_image_token_count(processor, resolution[0], resolution[1])
                    flat_image_token_counts.append(resolution_token_cache[resolution])
                else:
                    flat_image_token_counts.append(None)
            logger.info(f"[动态分辨率模式] Token count 缓存命中，共计算 {len(resolution_token_cache)} 种分辨率")
        elif target_size is None and processor is not None:
            logger.info("[原始分辨率模式] 计算每张图片的 image token 数量...")
            resolution_cache = {}
            flat_image_token_counts = compute_image_token_counts_batch(
                processor, flat_image_paths, num_proc=num_proc, resolution_cache=resolution_cache
            )
            logger.info(f"[原始分辨率模式] 发现 {len(resolution_cache)} 种不同的分辨率")
        
        # Reorganize refs per sample
        valid_messages, valid_refs_list, valid_num_images, valid_token_counts_list = [], [], [], []
        ref_idx = 0
        for msg, count in zip(messages, image_counts):
            sample_refs = flat_image_refs[ref_idx:ref_idx + count]
            sample_token_counts = flat_image_token_counts[ref_idx:ref_idx + count] if flat_image_token_counts else None
            ref_idx += count
            
            if all(r is not None for r in sample_refs) and sample_refs:
                valid_messages.append(msg)
                valid_refs_list.append(sample_refs)
                valid_num_images.append(len(sample_refs))
                if sample_token_counts:
                    valid_token_counts_list.append(sample_token_counts)
        
        if not valid_messages:
            raise ValueError("No valid multi-view samples after archiving.")
        
        dataset_dict = {
            "messages": valid_messages,
            "image_refs": valid_refs_list,
            "image_bases": [[archive_base_key] * n for n in valid_num_images],
            "num_images": valid_num_images
        }
        if valid_token_counts_list:
            dataset_dict["image_token_counts"] = valid_token_counts_list
        
        return Dataset.from_dict(dataset_dict)
    
    else:
        return Dataset.from_dict({"messages": [d["messages"] for d in formatted_data]})


def tokenize_dataset(
    dataset: Dataset,
    mode: str,
    tokenizer,
    processor,
    model_name: str,
    max_length: int,
    num_proc: int,
    target_size: Optional[Tuple[int, int]],
    image_archives_root: Optional[Path],
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Dataset:
    """Tokenize dataset based on mode."""
    
    is_vl_mode = mode in ("panorama", "multi_view", "multi_view2", "multi_view3")
    
    if mode == "panorama":
        # Compute fixed image token count if using fixed resolution
        fixed_image_token_count = None
        if target_size is not None:
            logger.info(f"[优化模式] 预计算固定分辨率 {target_size[0]}x{target_size[1]} 的 image token 数量...")
            fixed_image_token_count = compute_image_token_count(processor, target_size[0], target_size[1])
            logger.info(f"[优化模式] 每张图像需要 {fixed_image_token_count} 个 <|image_pad|> tokens")
        
        original_count = len(dataset)
        logger.info(f"Tokenizing VL dataset with {num_proc} processes...")
        result = dataset.map(
            lambda examples: tokenize_vl_function(
                examples, model_name, max_length, min_pixels, max_pixels,
                archive_base_path=str(image_archives_root) if fixed_image_token_count is None else None,
                fixed_image_token_count=fixed_image_token_count,
                tokenizer=tokenizer if fixed_image_token_count is not None else None,
            ),
            batched=True, batch_size=100, num_proc=num_proc,
            remove_columns=["messages"],
            desc="Tokenizing VL" + (" (optimized)" if fixed_image_token_count else "")
        )
        # 过滤掉空样本（被丢弃的超长序列）并打印统计信息
        result = result.filter(lambda x: len(x["input_ids"]) > 0)
        discarded = original_count - len(result)
        if discarded > 0:
            logger.warning(f"[VL Mode] 因序列长度超过 {max_length} 被丢弃的样本数: {discarded}")
        return result
    
    elif mode in ("multi_view", "multi_view2", "multi_view3"):
        fixed_image_token_count = None
        use_dynamic_token_counts = "image_token_counts" in dataset.column_names
        
        if target_size is not None:
            logger.info(f"[固定分辨率模式] 预计算固定分辨率 {target_size[0]}x{target_size[1]} 的 image token 数量...")
            fixed_image_token_count = compute_image_token_count(processor, target_size[0], target_size[1])
            logger.info(f"[固定分辨率模式] 每张图像需要 {fixed_image_token_count} 个 <|image_pad|> tokens")
        elif not use_dynamic_token_counts:
            raise ValueError("multiview_resolution is required, or image_token_counts must be present")
        
        mode_desc = " (动态分辨率)" if use_dynamic_token_counts else ""
        original_count = len(dataset)
        logger.info(f"Tokenizing multi-view dataset with {num_proc} processes...{mode_desc}")
        result = dataset.map(
            lambda examples: tokenize_multiview_function(
                examples, tokenizer, max_length, fixed_image_token_count
            ),
            batched=True, batch_size=100, num_proc=num_proc,
            remove_columns=["messages"],
            desc="Tokenizing multi-view" + mode_desc
        )
        # 过滤掉空样本（被丢弃的超长序列）并打印统计信息
        result = result.filter(lambda x: len(x["input_ids"]) > 0)
        discarded = original_count - len(result)
        if discarded > 0:
            logger.warning(f"[Multi-view Mode] 因序列长度超过 {max_length} 被丢弃的样本数: {discarded}")
        return result
    
    else:
        original_count = len(dataset)
        logger.info(f"Tokenizing dataset with {num_proc} processes...")
        result = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, max_length),
            batched=True, batch_size=100, num_proc=num_proc,
            remove_columns=["messages"],
            desc="Tokenizing"
        )
        # 过滤掉空样本（被丢弃的超长序列）并打印统计信息
        result = result.filter(lambda x: len(x["input_ids"]) > 0)
        discarded = original_count - len(result)
        if discarded > 0:
            logger.warning(f"[Text Mode] 因序列长度超过 {max_length} 被丢弃的样本数: {discarded}")
        return result


def apply_packing_if_enabled(
    dataset: Dataset,
    packing: bool,
    max_length: int,
    pad_token_id: int,
    is_vl_mode: bool
) -> Dataset:
    """Apply sequence packing if enabled."""
    if not packing:
        return dataset
    
    logger.info(f"Packing sequences with max_length={max_length}...")
    packed = pack_sequences(dataset, max_length, pad_token_id, is_vl_mode)
    logger.info(f"Total samples after packing: {len(packed)}")
    if is_vl_mode:
        logger.info("VL packing enabled: position_ids will be 3D (M-RoPE format) during training")
    return packed


def save_dataset_and_config(
    dataset_dict: DatasetDict,
    output_path: Path,
    mode: str,
    model_name: str,
    is_vl_mode: bool,
    processor,
    tokenizer,
    image_archives_root: Optional[Path] = None,
    archive_base_key: Optional[str] = None,
):
    """Save dataset and related config files."""
    logger.info(f"Saving dataset to {output_path}")
    dataset_dict.save_to_disk(str(output_path))
    
    if mode in ("panorama", "multi_view", "multi_view2", "multi_view3") and image_archives_root and archive_base_key:
        write_archive_info(output_path, archive_base_key, image_archives_root)
    
    # Save tokenizer/processor config
    tokenizer_config_path = output_path / "tokenizer_config"
    if is_vl_mode:
        processor.save_pretrained(tokenizer_config_path)
    else:
        tokenizer.save_pretrained(tokenizer_config_path)
    
    # Save mode info
    mode_info_path = output_path / "mode_info.json"
    with open(mode_info_path, "w") as f:
        json.dump({"mode": mode, "model_name": model_name, "is_vl": is_vl_mode}, f)


# =============================================================================
# Main Entry Points
# =============================================================================

def prepare_dataset_simple(
    json_dir: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-8B",
    max_length: int = 2048,
    test_split: float = 0.1,
    seed: int = 42,
    num_proc: int = 8,
    merge_mode: bool = False,
    mode: str = "unconditional",
    packing: bool = False,
    caption_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    image_resolution: Optional[str] = None,
    archive_shard_size: Optional[int] = None,
    num_views_range: Optional[tuple] = None,
    multiview_resolution: Optional[str] = None,
    multiview_resolution_min: Optional[str] = None,
    multiview_resolution_max: Optional[str] = None,
    image_format: str = "png",
    image_quality: int = 85
) -> None:
    """
    Prepare and save dataset for fine-tuning (simple mode - one-time processing).
    Suitable for small to medium datasets or systems with sufficient memory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    is_vl_mode = mode in ("panorama", "multi_view", "multi_view2", "multi_view3")
    is_multi_view = mode in ("multi_view", "multi_view2", "multi_view3")
    
    # Parse resolutions
    target_size = parse_resolution(image_resolution) if mode == "panorama" else None
    multiview_target_size = parse_resolution(multiview_resolution) if is_multi_view else None
    
    # Parse dynamic resolution range for multi-view
    multiview_min_size = parse_resolution(multiview_resolution_min) if is_multi_view else None
    multiview_max_size = parse_resolution(multiview_resolution_max) if is_multi_view else None
    use_dynamic_resolution = is_multi_view and multiview_min_size is not None and multiview_max_size is not None
    
    # Compute pixel limits for panorama
    min_pixels, max_pixels = None, None
    if mode == "panorama" and target_size:
        max_pixels = target_size[0] * target_size[1]
        min_pixels = max_pixels // 2
        logger.info(f"Image resolution: {target_size[0]}x{target_size[1]}")
    
    if is_multi_view and multiview_target_size:
        logger.info(f"Multi-view resolution: {multiview_target_size[0]}x{multiview_target_size[1]}")
    
    if use_dynamic_resolution:
        logger.info(f"Multi-view dynamic resolution: {multiview_min_size} ~ {multiview_max_size}")
    
    # Load existing dataset for merge mode
    existing_dataset = None
    if merge_mode and (output_path / "dataset_dict.json").exists():
        logger.info(f"Merge mode: Loading existing dataset from {output_dir}")
        try:
            existing_dataset = DatasetDict.load_from_disk(output_dir)
            logger.info(f"Loaded existing dataset - Train: {len(existing_dataset['train'])} samples")
        except Exception as e:
            logger.warning(f"Failed to load existing dataset: {e}")
    
    # Load tokenizer/processor
    processor, tokenizer = load_tokenizer_or_processor(model_name, is_vl_mode)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup image archives
    image_archives_root = None
    archive_base_key = None
    existing_archive_info = None
    if is_vl_mode:
        existing_archive_info = read_archive_info(output_path) if merge_mode else None
        image_archives_root = ensure_archives_root(output_path, merge_mode, existing_archive_info)
        archive_base_key = existing_archive_info.get("base_key") if existing_archive_info else uuid.uuid4().hex
    
    # Load data
    all_data = load_data_by_mode(mode, json_dir, caption_dir, image_dir, num_views_range)
    
    # Convert to Qwen format
    logger.info("Converting to Qwen format...")
    formatted_data = convert_to_qwen_format(all_data, mode=mode)
    
    # Create dataset
    effective_target_size = target_size if mode == "panorama" else (multiview_target_size if not use_dynamic_resolution else None)
    shard_size = archive_shard_size or 1000
    dataset = create_dataset_from_formatted_data(
        formatted_data, mode, image_archives_root, archive_base_key,
        effective_target_size, num_proc, shard_size, processor,
        multiview_min_size if use_dynamic_resolution else None,
        multiview_max_size if use_dynamic_resolution else None,
        image_format, image_quality
    )
    logger.info(f"Total samples before tokenization: {len(dataset)}")
    
    # Tokenize
    tokenized_dataset = tokenize_dataset(
        dataset, mode, tokenizer, processor, model_name, max_length, num_proc,
        effective_target_size, image_archives_root, min_pixels, max_pixels
    )
    logger.info(f"Total samples after tokenization: {len(tokenized_dataset)}")
    
    # Apply packing
    tokenized_dataset = apply_packing_if_enabled(
        tokenized_dataset, packing, max_length, tokenizer.pad_token_id, is_vl_mode
    )
    
    # Merge or split
    if merge_mode and existing_dataset is not None:
        logger.info("Merging new data with existing dataset...")
        new_dict = tokenized_dataset.train_test_split(test_size=test_split, seed=seed) if test_split > 0 else DatasetDict({"train": tokenized_dataset})
        
        merged_train = concatenate_datasets([existing_dataset['train'], new_dict['train']])
        if 'test' in existing_dataset and 'test' in new_dict:
            merged_test = concatenate_datasets([existing_dataset['test'], new_dict['test']])
            dataset_dict = DatasetDict({"train": merged_train, "test": merged_test})
        else:
            dataset_dict = DatasetDict({"train": merged_train})
    else:
        dataset_dict = tokenized_dataset.train_test_split(test_size=test_split, seed=seed) if test_split > 0 else DatasetDict({"train": tokenized_dataset})
    
    # Save
    save_dataset_and_config(
        dataset_dict, output_path, mode, model_name, is_vl_mode,
        processor, tokenizer, image_archives_root, archive_base_key
    )
    
    logger.info("Dataset preparation completed")
    logger.info(f"Train samples: {len(dataset_dict['train'])}")
    if 'test' in dataset_dict:
        logger.info(f"Test samples: {len(dataset_dict['test'])}")


def prepare_dataset_batch(
    json_dir: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-8B",
    max_length: int = 2048,
    test_split: float = 0.1,
    seed: int = 42,
    num_proc: int = 8,
    batch_size: int = 5000,
    merge_mode: bool = False,
    mode: str = "unconditional",
    packing: bool = False,
    caption_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    batch_index: Optional[int] = None,
    total_batches: Optional[int] = None,
    image_resolution: Optional[str] = None,
    archive_shard_size: Optional[int] = None,
    num_views_range: Optional[tuple] = None,
    multiview_resolution: Optional[str] = None,
    multiview_resolution_min: Optional[str] = None,
    multiview_resolution_max: Optional[str] = None,
    image_format: str = "png",
    image_quality: int = 85
) -> None:
    """
    Prepare and save dataset for fine-tuning (batch mode - memory efficient).
    Processes large datasets in batches to avoid memory issues.
    """
    import random
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    is_vl_mode = mode in ("panorama", "multi_view", "multi_view2", "multi_view3")
    is_multi_view = mode in ("multi_view", "multi_view2", "multi_view3")
    
    # Parse resolutions
    target_size = parse_resolution(image_resolution) if mode == "panorama" else None
    multiview_target_size = parse_resolution(multiview_resolution) if is_multi_view else None
    
    # Parse dynamic resolution range for multi-view
    multiview_min_size = parse_resolution(multiview_resolution_min) if is_multi_view else None
    multiview_max_size = parse_resolution(multiview_resolution_max) if is_multi_view else None
    use_dynamic_resolution = is_multi_view and multiview_min_size is not None and multiview_max_size is not None
    
    if use_dynamic_resolution:
        logger.info(f"Multi-view dynamic resolution: {multiview_min_size} ~ {multiview_max_size}")
    
    # Compute pixel limits
    min_pixels, max_pixels = None, None
    if mode == "panorama" and target_size:
        max_pixels = target_size[0] * target_size[1]
        min_pixels = max_pixels // 2
    
    # Setup archives root
    archives_root: Optional[Path] = None
    if is_vl_mode:
        archives_root = output_path / "image_archives"
        if batch_index is None and archives_root.exists():
            shutil.rmtree(archives_root)
        archives_root.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer/processor
    processor, tokenizer = load_tokenizer_or_processor(model_name, is_vl_mode)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get and filter JSON files
    json_path = Path(json_dir)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    
    json_files = sorted(list(json_path.glob("*.json")))
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    # Filter based on mode
    if mode == "caption":
        if not caption_dir:
            raise ValueError("caption_dir is required for 'caption' mode")
        caption_stems = {f.stem for f in Path(caption_dir).glob("*.txt")}
        json_files = [f for f in json_files if f.stem in caption_stems]
    elif mode == "panorama":
        if not image_dir:
            raise ValueError("image_dir is required for 'panorama' mode")
        valid_files = []
        for f in tqdm(json_files, desc="Matching panorama images"):
            if (Path(image_dir) / f.stem / "panorama" / "panorama_rgb.png").exists():
                valid_files.append(f)
        json_files = valid_files
    elif is_multi_view:
        if not image_dir:
            raise ValueError("image_dir is required for multi-view modes")
        if mode == "multi_view3":
            # multi_view3: 仅需 front.png 存在
            valid_files = []
            for f in tqdm(json_files, desc="Matching front-view images"):
                front_path = Path(image_dir) / f.stem / "front.png"
                if front_path.exists():
                    valid_files.append(f)
            json_files = valid_files
        else:
            min_views = (num_views_range or (4, 8))[0]
            subdir = "orbit" if mode == "multi_view" else None
            pattern = "orbit_*.png" if mode == "multi_view" else "*.png"
            valid_files = []
            for f in tqdm(json_files, desc="Matching multi-view images"):
                candidate = Path(image_dir) / f.stem
                if subdir:
                    candidate = candidate / subdir
                if candidate.exists() and len(list(candidate.glob(pattern))) >= min_views:
                    valid_files.append(f)
            json_files = valid_files
    
    total_files = len(json_files)
    num_batches = total_batches or ((total_files + batch_size - 1) // batch_size)
    
    logger.info(f"Found {total_files} JSON files to process")
    logger.info(f"[BATCH MODE] Processing in batches of {batch_size}, total batches: {num_batches}")
    
    # Determine batches to process
    batches_to_process = [batch_index] if batch_index is not None else list(range(num_batches))
    
    # Pre-compute image token count
    fixed_image_token_count = None
    effective_target_size = target_size if mode == "panorama" else multiview_target_size
    if effective_target_size is not None and is_vl_mode:
        logger.info(f"[优化模式] 预计算固定分辨率 {effective_target_size[0]}x{effective_target_size[1]} 的 image token 数量...")
        fixed_image_token_count = compute_image_token_count(processor, effective_target_size[0], effective_target_size[1])
        logger.info(f"[优化模式] 每张图像需要 {fixed_image_token_count} 个 <|image_pad|> tokens")
    
    total_samples = 0
    
    for batch_idx in batches_to_process:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        
        if start_idx >= total_files:
            logger.warning(f"Batch {batch_idx} out of range, skipping")
            continue
        
        batch_files = json_files[start_idx:end_idx]
        logger.info(f"Processing batch {batch_idx}/{num_batches - 1}: files {start_idx} to {end_idx - 1}")
        
        # Load batch data
        batch_data = []
        for json_file in tqdm(batch_files, desc=f"Loading batch {batch_idx}"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                if mode == "caption":
                    caption_file = Path(caption_dir) / f"{json_file.stem}.txt"
                    with open(caption_file, 'r', encoding='utf-8') as cf:
                        caption = cf.read().strip()
                    batch_data.append({"caption": caption, "json_data": content, "filename": json_file.stem})
                elif mode == "panorama":
                    panorama_path = Path(image_dir) / json_file.stem / "panorama" / "panorama_rgb.png"
                    batch_data.append({"image_path": str(panorama_path), "json_data": content, "filename": json_file.stem})
                elif is_multi_view:
                    if mode == "multi_view3":
                        # multi_view3: 仅使用 front.png 单个视角
                        front_path = Path(image_dir) / json_file.stem / "front.png"
                        batch_data.append({"image_paths": [str(front_path)], "json_data": content, "filename": json_file.stem})
                    else:
                        view_dir = Path(image_dir) / json_file.stem
                        if mode == "multi_view":
                            view_dir = view_dir / "orbit"
                        pattern = "orbit_*.png" if mode == "multi_view" else "*.png"
                        view_images = sorted(view_dir.glob(pattern))
                        views_range = num_views_range or (4, 8)
                        n = random.randint(views_range[0], min(views_range[1], len(view_images)))
                        selected = random.sample(view_images, n)
                        random.shuffle(selected)
                        batch_data.append({"image_paths": [str(p) for p in selected], "json_data": content, "filename": json_file.stem})
                else:
                    batch_data.append(content)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        if not batch_data:
            continue
        
        # Convert format
        formatted_batch = convert_to_qwen_format(batch_data, mode=mode)
        
        # Create dataset
        subdir_name = f"batch_{batch_idx:04d}"
        batch_archive_key = f"{subdir_name}_{uuid.uuid4().hex}"
        
        if is_vl_mode:
            subdir_path = archives_root / subdir_name
            if subdir_path.exists():
                shutil.rmtree(subdir_path)
        
        if mode == "panorama":
            messages = [item["messages"] for item in formatted_batch]
            image_paths = [item.get("image_path") for item in formatted_batch]
            shard_size = archive_shard_size or max(1, len(image_paths))
            
            image_refs = create_image_archives(
                image_paths, archives_root, target_size, num_proc, shard_size, subdir_name,
                image_format, image_quality
            )
            
            valid_messages, valid_refs = [], []
            for msg, ref in zip(messages, image_refs):
                if ref is not None:
                    valid_messages.append(msg)
                    valid_refs.append(ref)
            
            if not valid_messages:
                continue
            
            batch_dataset = Dataset.from_dict({
                "messages": valid_messages,
                "image_ref": valid_refs,
                "image_base": [batch_archive_key] * len(valid_refs)
            })
        elif is_multi_view:
            messages = [item["messages"] for item in formatted_batch]
            all_image_paths = [item.get("image_paths", []) for item in formatted_batch]
            
            flat_paths, image_counts = [], []
            for paths in all_image_paths:
                flat_paths.extend(paths)
                image_counts.append(len(paths))
            
            shard_size = archive_shard_size or max(1, len(flat_paths))
            
            # Check if using dynamic resolution mode
            flat_image_resolutions = None
            if use_dynamic_resolution:
                flat_refs, flat_image_resolutions = create_image_archives_dynamic(
                    flat_paths, archives_root, multiview_min_size, multiview_max_size, 
                    num_proc, shard_size, subdir_name, image_format, image_quality
                )
            else:
                flat_refs = create_image_archives(
                    flat_paths, archives_root, multiview_target_size, num_proc, shard_size, subdir_name,
                    image_format, image_quality
                )
            
            # Token counts
            flat_token_counts = None
            if use_dynamic_resolution:
                # For dynamic resolution, compute token counts based on actual resolutions
                # 使用缓存避免重复计算相同分辨率的 token count
                resolution_token_cache = {}
                flat_token_counts = []
                unique_resolutions = set(r for r in flat_image_resolutions if r is not None)
                logger.info(f"[动态分辨率模式] 发现 {len(unique_resolutions)} 种不同的分辨率，计算 token counts...")
                for resolution in tqdm(flat_image_resolutions, desc=f"Computing token counts (batch {batch_idx})"):
                    if resolution is not None:
                        if resolution not in resolution_token_cache:
                            resolution_token_cache[resolution] = compute_image_token_count(processor, resolution[0], resolution[1])
                        flat_token_counts.append(resolution_token_cache[resolution])
                    else:
                        flat_token_counts.append(None)
                logger.info(f"[动态分辨率模式] Token count 缓存命中，共计算 {len(resolution_token_cache)} 种分辨率")
            elif multiview_target_size is None:
                resolution_cache = {}
                flat_token_counts = compute_image_token_counts_batch(processor, flat_paths, num_proc, resolution_cache)
            
            valid_messages, valid_refs_list, valid_num_images, valid_token_counts = [], [], [], []
            ref_idx = 0
            for msg, count in zip(messages, image_counts):
                sample_refs = flat_refs[ref_idx:ref_idx + count]
                sample_counts = flat_token_counts[ref_idx:ref_idx + count] if flat_token_counts else None
                ref_idx += count
                
                if all(r is not None for r in sample_refs) and sample_refs:
                    valid_messages.append(msg)
                    valid_refs_list.append(sample_refs)
                    valid_num_images.append(len(sample_refs))
                    if sample_counts:
                        valid_token_counts.append(sample_counts)
            
            if not valid_messages:
                continue
            
            dataset_dict = {
                "messages": valid_messages,
                "image_refs": valid_refs_list,
                "image_bases": [[batch_archive_key] * n for n in valid_num_images],
                "num_images": valid_num_images
            }
            if valid_token_counts:
                dataset_dict["image_token_counts"] = valid_token_counts
            batch_dataset = Dataset.from_dict(dataset_dict)
        else:
            batch_dataset = Dataset.from_dict({"messages": [d["messages"] for d in formatted_batch]})
        
        # Tokenize
        original_batch_count = len(batch_dataset)
        
        if mode == "panorama":
            tokenized_batch = batch_dataset.map(
                lambda ex: tokenize_vl_function(
                    ex, model_name, max_length, min_pixels, max_pixels,
                    archive_base_path=str(archives_root) if fixed_image_token_count is None else None,
                    fixed_image_token_count=fixed_image_token_count,
                    tokenizer=tokenizer if fixed_image_token_count is not None else None,
                ),
                batched=True, batch_size=100, num_proc=num_proc, remove_columns=["messages"],
                desc=f"Tokenizing VL batch {batch_idx}"
            )
            # 过滤掉空样本（被丢弃的超长序列）并打印统计信息
            tokenized_batch = tokenized_batch.filter(lambda x: len(x["input_ids"]) > 0)
            discarded = original_batch_count - len(tokenized_batch)
            if discarded > 0:
                logger.warning(f"[Batch {batch_idx} VL Mode] 因序列长度超过 {max_length} 被丢弃的样本数: {discarded}")
        elif is_multi_view:
            use_dynamic = "image_token_counts" in batch_dataset.column_names
            if fixed_image_token_count is None and not use_dynamic:
                raise ValueError("multiview_resolution required or image_token_counts must be present")
            
            tokenized_batch = batch_dataset.map(
                lambda ex: tokenize_multiview_function(ex, tokenizer, max_length, fixed_image_token_count),
                batched=True, batch_size=100, num_proc=num_proc, remove_columns=["messages"],
                desc=f"Tokenizing multi-view batch {batch_idx}"
            )
            # 过滤掉空样本（被丢弃的超长序列）并打印统计信息
            tokenized_batch = tokenized_batch.filter(lambda x: len(x["input_ids"]) > 0)
            discarded = original_batch_count - len(tokenized_batch)
            if discarded > 0:
                logger.warning(f"[Batch {batch_idx} Multi-view Mode] 因序列长度超过 {max_length} 被丢弃的样本数: {discarded}")
        else:
            tokenized_batch = batch_dataset.map(
                lambda ex: tokenize_function(ex, tokenizer, max_length),
                batched=True, batch_size=100, num_proc=num_proc, remove_columns=["messages"],
                desc=f"Tokenizing batch {batch_idx}"
            )
            # 过滤掉空样本（被丢弃的超长序列）并打印统计信息
            tokenized_batch = tokenized_batch.filter(lambda x: len(x["input_ids"]) > 0)
            discarded = original_batch_count - len(tokenized_batch)
            if discarded > 0:
                logger.warning(f"[Batch {batch_idx} Text Mode] 因序列长度超过 {max_length} 被丢弃的样本数: {discarded}")
        
        # Apply packing
        tokenized_batch = apply_packing_if_enabled(
            tokenized_batch, packing, max_length, tokenizer.pad_token_id, is_vl_mode
        )
        
        # Save batch
        batch_dir = output_path / f"batch_{batch_idx:04d}"
        tokenized_batch.save_to_disk(str(batch_dir))
        if is_vl_mode:
            write_archive_info(batch_dir, batch_archive_key, archives_root)
        
        batch_samples = len(tokenized_batch)
        total_samples += batch_samples
        avg_seq_len = sum(len(x) for x in tokenized_batch["input_ids"]) / batch_samples if batch_samples > 0 else 0
        logger.info(f"Batch {batch_idx} completed: {batch_samples} samples, avg sequence length: {avg_seq_len:.1f}")
        
        # Clean up
        del batch_data, formatted_batch, batch_dataset, tokenized_batch
        gc.collect()
    
    # Save config (only for batch 0 or full run)
    if batch_index is None or batch_index == 0:
        tokenizer_config_path = output_path / "tokenizer_config"
        if is_vl_mode:
            processor.save_pretrained(tokenizer_config_path)
        else:
            tokenizer.save_pretrained(tokenizer_config_path)
        
        mode_info_path = output_path / "mode_info.json"
        with open(mode_info_path, "w") as f:
            json.dump({"mode": mode, "model_name": model_name, "is_vl": is_vl_mode}, f)
    
    logger.info("Dataset preparation completed")
    logger.info(f"Total samples: {total_samples}")


def prepare_dataset(
    json_dir: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-8B",
    max_length: int = 2048,
    test_split: float = 0.1,
    seed: int = 42,
    num_proc: int = 8,
    batch_mode: bool = False,
    batch_size: int = 5000,
    merge_mode: bool = False,
    mode: str = "unconditional",
    packing: bool = False,
    caption_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    batch_index: Optional[int] = None,
    total_batches: Optional[int] = None,
    image_resolution: Optional[str] = None,
    archive_shard_size: Optional[int] = None,
    num_views_range: Optional[tuple] = None,
    multiview_resolution: Optional[str] = None,
    multiview_resolution_min: Optional[str] = None,
    multiview_resolution_max: Optional[str] = None,
    image_format: str = "png",
    image_quality: int = 85
) -> None:
    """
    Prepare and save dataset for fine-tuning.
    
    Args:
        json_dir: Directory containing JSON files
        output_dir: Directory to save the processed dataset
        model_name: HuggingFace model name for tokenizer
        max_length: Maximum sequence length
        test_split: Proportion of data for test set
        seed: Random seed for splitting
        num_proc: Number of processes for parallel processing
        batch_mode: If True, use batch processing mode (memory efficient)
        batch_size: Files per batch (only for batch_mode)
        merge_mode: If True, merge with existing dataset
        mode: Generation mode
        packing: If True, pack sequences together
        caption_dir: Caption files directory (for caption mode)
        image_dir: Image directory (for VL modes)
        batch_index: Process only this batch (for parallel batch processing)
        total_batches: Total batch count (for parallel batch processing)
        image_resolution: Panorama resolution 'WIDTHxHEIGHT'
        archive_shard_size: Max images per tar shard
        num_views_range: (min_views, max_views) for multi-view modes
        multiview_resolution: Multi-view image resolution 'WIDTHxHEIGHT'
        multiview_resolution_min: Min resolution for dynamic augmentation 'WIDTHxHEIGHT'
        multiview_resolution_max: Max resolution for dynamic augmentation 'WIDTHxHEIGHT'
        image_format: Image format ('png' or 'jpeg')
        image_quality: JPEG quality (1-100), ignored for png
    """
    if batch_mode:
        logger.info("Using BATCH MODE (memory efficient)")
        prepare_dataset_batch(
            json_dir=json_dir, output_dir=output_dir, model_name=model_name,
            max_length=max_length, test_split=test_split, seed=seed,
            num_proc=num_proc, batch_size=batch_size, merge_mode=merge_mode,
            mode=mode, packing=packing, caption_dir=caption_dir, image_dir=image_dir,
            batch_index=batch_index, total_batches=total_batches,
            image_resolution=image_resolution, archive_shard_size=archive_shard_size,
            num_views_range=num_views_range, multiview_resolution=multiview_resolution,
            multiview_resolution_min=multiview_resolution_min,
            multiview_resolution_max=multiview_resolution_max,
            image_format=image_format, image_quality=image_quality
        )
    else:
        logger.info("Using SIMPLE MODE (one-time processing)")
        prepare_dataset_simple(
            json_dir=json_dir, output_dir=output_dir, model_name=model_name,
            max_length=max_length, test_split=test_split, seed=seed,
            num_proc=num_proc, merge_mode=merge_mode, mode=mode, packing=packing,
            caption_dir=caption_dir, image_dir=image_dir,
            image_resolution=image_resolution, archive_shard_size=archive_shard_size,
            num_views_range=num_views_range, multiview_resolution=multiview_resolution,
            multiview_resolution_min=multiview_resolution_min,
            multiview_resolution_max=multiview_resolution_max,
            image_format=image_format, image_quality=image_quality
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for fine-tuning with multi-processing support"
    )
    parser.add_argument("--json_dir", type=str, required=True, help="Directory containing JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-8B", help="HuggingFace model name")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test set proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes")
    parser.add_argument("--batch_mode", action="store_true", help="Enable batch processing mode")
    parser.add_argument("--batch_size", type=int, default=5000, help="Files per batch (batch_mode only)")
    parser.add_argument("--merge_mode", action="store_true", help="Merge with existing dataset")
    parser.add_argument(
        "--mode", type=str, default="unconditional",
        choices=["unconditional", "blueprint", "caption", "panorama", "multi_view", "multi_view2", "multi_view3", "contour"],
        help="Generation mode"
    )
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing")
    parser.add_argument("--caption_dir", type=str, default=None, help="Caption files directory")
    parser.add_argument("--image_dir", type=str, default=None, help="Image directory")
    parser.add_argument("--batch_index", type=int, default=None, help="Process only this batch index")
    parser.add_argument("--total_batches", type=int, default=None, help="Total batch count")
    parser.add_argument("--image_resolution", type=str, default="1024x512", help="Panorama resolution")
    parser.add_argument("--archive_shard_size", type=int, default=None, help="Max images per tar shard")
    parser.add_argument("--num_views_min", type=int, default=4, help="Min views for multi-view modes")
    parser.add_argument("--num_views_max", type=int, default=8, help="Max views for multi-view modes")
    parser.add_argument("--multiview_resolution", type=str, default=None, help="Multi-view image resolution")
    parser.add_argument("--multiview_resolution_min", type=str, default=None, 
                        help="Min resolution for dynamic augmentation (e.g., 512x512)")
    parser.add_argument("--multiview_resolution_max", type=str, default=None, 
                        help="Max resolution for dynamic augmentation (e.g., 1024x1024)")
    parser.add_argument("--image_format", type=str, default="png", choices=["png", "jpeg"],
                        help="Image format for archiving: 'png' (lossless) or 'jpeg' (lossy, smaller)")
    parser.add_argument("--image_quality", type=int, default=85,
                        help="JPEG quality (1-100), only used when image_format='jpeg'")
    
    args = parser.parse_args()
    
    # Validation
    if args.mode == "caption" and not args.caption_dir:
        parser.error("--caption_dir required for caption mode")
    if args.mode == "panorama" and not args.image_dir:
        parser.error("--image_dir required for panorama mode")
    if args.mode in ("multi_view", "multi_view2", "multi_view3") and not args.image_dir:
        parser.error("--image_dir required for multi-view modes")
    if args.batch_index is not None and not args.batch_mode:
        parser.error("--batch_index requires --batch_mode")
    
    num_views_range = (args.num_views_min, args.num_views_max) if args.mode in ("multi_view", "multi_view2") else None
    
    prepare_dataset(
        json_dir=args.json_dir, output_dir=args.output_dir, model_name=args.model_name,
        max_length=args.max_length, test_split=args.test_split, seed=args.seed,
        num_proc=args.num_proc, batch_mode=args.batch_mode, batch_size=args.batch_size,
        merge_mode=args.merge_mode, mode=args.mode, packing=args.packing,
        caption_dir=args.caption_dir, image_dir=args.image_dir,
        batch_index=args.batch_index, total_batches=args.total_batches,
        image_resolution=args.image_resolution, archive_shard_size=args.archive_shard_size,
        num_views_range=num_views_range, multiview_resolution=args.multiview_resolution,
        multiview_resolution_min=args.multiview_resolution_min,
        multiview_resolution_max=args.multiview_resolution_max,
        image_format=args.image_format, image_quality=args.image_quality
    )


if __name__ == "__main__":
    main()
