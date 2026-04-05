"""
Sequence packing utilities to reduce padding waste in training.
"""

from typing import List, Optional
from datasets import Dataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("packing")


def pack_sequences(dataset, max_length: int, pad_token_id: int, is_vl_mode: bool = False) -> Dataset:
    """
    Pack multiple short sequences into longer ones to reduce padding waste.
    
    Uses a greedy first-fit algorithm to pack sequences efficiently.
    Each packed sequence contains multiple original sequences separated by boundaries.
    
    This packing mechanism leverages Flash Attention 2's flash_attn_varlen_func:
    - The 'sequence_lengths' field stores the length of each sub-sequence
    - During training, position_ids are generated with each sub-sequence starting from 0
    - Flash Attention detects these position resets and isolates attention between sub-sequences
    
    For VL models (Qwen2-VL/Qwen3-VL), the position_ids will be 3D (M-RoPE format).
    
    Args:
        dataset: HuggingFace dataset with 'input_ids' and 'labels' columns
        max_length: Maximum length for packed sequences
        pad_token_id: Padding token ID (used as separator between sequences)
        is_vl_mode: Whether this is for VL model (preserves image-related fields)
    
    Returns:
        New dataset with packed sequences and 'sequence_lengths' column for attention isolation
    """
    # Sort by length (descending) for better packing efficiency
    lengths = [len(x) for x in dataset["input_ids"]]
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    
    packed_input_ids = []
    packed_labels = []
    packed_sequence_lengths = []
    
    # Check for image-related fields
    has_single_image_refs = is_vl_mode and "image_ref" in dataset.column_names
    has_single_image_bases = is_vl_mode and "image_base" in dataset.column_names
    has_multi_image_refs = is_vl_mode and "image_refs" in dataset.column_names
    has_multi_image_bases = is_vl_mode and "image_bases" in dataset.column_names
    has_image_token_counts = is_vl_mode and "image_token_counts" in dataset.column_names
    
    has_image_refs = has_single_image_refs or has_multi_image_refs
    has_image_bases = has_single_image_bases or has_multi_image_bases
    packed_image_refs = [] if has_image_refs else None
    packed_image_bases = [] if has_image_bases else None
    packed_image_token_counts = [] if has_image_token_counts else None
    packed_has_image = [] if is_vl_mode else None
    
    # Greedy first-fit packing
    bins = []
    
    for idx in sorted_indices:
        input_ids = dataset["input_ids"][idx]
        labels = dataset["labels"][idx]
        seq_len = len(input_ids)
        
        # Get image data for VL mode
        image_refs_list = []
        image_bases_list = []
        image_token_counts_list = []
        has_img = False
        if is_vl_mode:
            if has_multi_image_refs:
                refs = dataset["image_refs"][idx]
                if refs:
                    image_refs_list = refs if isinstance(refs, list) else [refs]
                    has_img = len(image_refs_list) > 0
            elif has_single_image_refs:
                ref = dataset["image_ref"][idx]
                if ref:
                    image_refs_list = [ref]
                    has_img = True
            
            if has_multi_image_bases:
                bases = dataset["image_bases"][idx]
                if bases:
                    image_bases_list = bases if isinstance(bases, list) else [bases]
            elif has_single_image_bases:
                base = dataset["image_base"][idx]
                if base:
                    image_bases_list = [base]
            
            if has_image_token_counts:
                counts = dataset["image_token_counts"][idx]
                if counts:
                    image_token_counts_list = counts if isinstance(counts, list) else [counts]
            
            if not has_img and "has_image" in dataset.column_names:
                has_img = dataset["has_image"][idx]
        
        # Skip sequences longer than max_length
        if seq_len > max_length:
            packed_input_ids.append(input_ids[:max_length])
            packed_labels.append(labels[:max_length])
            packed_sequence_lengths.append([min(seq_len, max_length)])
            if is_vl_mode:
                if packed_image_refs is not None:
                    packed_image_refs.append(image_refs_list if image_refs_list else [])
                if packed_image_bases is not None:
                    packed_image_bases.append(image_bases_list if image_bases_list else [])
                if packed_image_token_counts is not None:
                    packed_image_token_counts.append(image_token_counts_list if image_token_counts_list else [])
                packed_has_image.append([has_img])
            continue
        
        # Find first bin that can fit this sequence
        placed = False
        for bin_data in bins:
            if bin_data["total_len"] + seq_len <= max_length:
                bin_data["input_ids"].extend(input_ids)
                bin_data["labels"].extend(labels)
                bin_data["lengths"].append(seq_len)
                bin_data["total_len"] += seq_len
                if is_vl_mode:
                    if packed_image_refs is not None:
                        bin_data.setdefault("image_refs", []).extend(image_refs_list)
                    if packed_image_bases is not None:
                        bin_data.setdefault("image_bases", []).extend(image_bases_list)
                    if packed_image_token_counts is not None:
                        bin_data.setdefault("image_token_counts", []).extend(image_token_counts_list)
                    bin_data.setdefault("has_image", []).append(has_img)
                placed = True
                break
        
        # Create new bin if needed
        if not placed:
            new_bin = {
                "input_ids": list(input_ids),
                "labels": list(labels),
                "lengths": [seq_len],
                "total_len": seq_len
            }
            if is_vl_mode:
                if packed_image_refs is not None:
                    new_bin["image_refs"] = list(image_refs_list)
                if packed_image_bases is not None:
                    new_bin["image_bases"] = list(image_bases_list)
                if packed_image_token_counts is not None:
                    new_bin["image_token_counts"] = list(image_token_counts_list)
                new_bin["has_image"] = [has_img]
            bins.append(new_bin)
    
    # Convert bins to final format
    for bin_data in bins:
        packed_input_ids.append(bin_data["input_ids"])
        packed_labels.append(bin_data["labels"])
        packed_sequence_lengths.append(bin_data["lengths"])
        if is_vl_mode:
            if packed_image_refs is not None:
                packed_image_refs.append(bin_data.get("image_refs", []))
            if packed_image_bases is not None:
                packed_image_bases.append(bin_data.get("image_bases", []))
            if packed_image_token_counts is not None:
                packed_image_token_counts.append(bin_data.get("image_token_counts", []))
            packed_has_image.append(bin_data.get("has_image", []))
    
    logger.info(f"Packing complete: {len(dataset)} sequences -> {len(packed_input_ids)} packed sequences")
    logger.info(f"Average sequences per pack: {len(dataset) / len(packed_input_ids):.2f}")
    
    result_dict = {
        "input_ids": packed_input_ids,
        "labels": packed_labels,
        "sequence_lengths": packed_sequence_lengths
    }
    
    if is_vl_mode:
        result_dict["has_image"] = packed_has_image
        if packed_image_refs is not None:
            result_dict["image_refs"] = packed_image_refs
        if packed_image_bases is not None:
            result_dict["image_bases"] = packed_image_bases
        if packed_image_token_counts is not None:
            result_dict["image_token_counts"] = packed_image_token_counts
    
    return Dataset.from_dict(result_dict)
