"""
Tokenization utilities for text and vision-language models.
"""

from typing import Dict, List, Any, Optional
from PIL import Image
from multiprocessing import Pool

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger
from src.data.image_archives import (
    load_image_from_archive_ref,
    get_image_resolution,
    get_image_resolutions_batch,
)

logger = setup_logger("tokenizers")


def compute_image_token_count(processor, width: int, height: int) -> int:
    """
    Pre-compute the number of <|image_pad|> tokens for a fixed resolution image.
    
    This value is fixed for the same resolution and can be cached for reuse.
    
    Args:
        processor: Qwen-VL processor
        width: Image width
        height: Image height
    
    Returns:
        int: Number of <|image_pad|> tokens
    """
    dummy_image = Image.new('RGB', (width, height), color='red')
    
    test_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": dummy_image},
            {"type": "text", "text": "test"}
        ]
    }]
    
    text = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[dummy_image], return_tensors=None, padding=False)
    input_ids = inputs["input_ids"][0]
    
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    num_image_tokens = sum(1 for tid in input_ids if tid == image_token_id)
    
    return num_image_tokens


def compute_image_token_count_for_resolution(processor, resolution: tuple) -> int:
    """
    Compute the number of <|image_pad|> tokens for a given resolution.
    
    Args:
        processor: Qwen-VL processor
        resolution: (width, height) tuple
    
    Returns:
        int: Number of <|image_pad|> tokens
    """
    return compute_image_token_count(processor, resolution[0], resolution[1])


def compute_image_token_counts_batch(
    processor,
    image_paths: List[str],
    num_proc: int = 1,
    resolution_cache: Optional[Dict[tuple, int]] = None
) -> List[int]:
    """
    Batch compute token counts for each image based on original resolution.
    
    Uses resolution cache to avoid recomputing for same resolutions.
    
    Args:
        processor: Qwen-VL processor
        image_paths: List of image file paths
        num_proc: Number of parallel processes
        resolution_cache: Optional resolution->token_count cache dict
    
    Returns:
        List[int]: Token count for each image
    """
    if resolution_cache is None:
        resolution_cache = {}
    
    resolutions = get_image_resolutions_batch(image_paths, num_proc)
    
    token_counts = []
    for res in resolutions:
        if res is None:
            token_counts.append(256)  # fallback default
        elif res in resolution_cache:
            token_counts.append(resolution_cache[res])
        else:
            count = compute_image_token_count_for_resolution(processor, res)
            resolution_cache[res] = count
            token_counts.append(count)
    
    return token_counts


def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenization function for use with datasets.map().
    
    Properly masks user input tokens by setting their labels to -100,
    so that loss is only computed on assistant responses.
    
    If sequence length exceeds max_length after tokenization, the sample is discarded
    (returns empty lists for that sample, which should be filtered out later).
    
    Args:
        examples: Batch of examples with 'messages' field
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with tokenized data
    """
    input_ids_batch = []
    labels_batch = []
    
    for messages in examples["messages"]:
        input_ids = []
        labels = []
        
        for i, message in enumerate(messages):
            role = message["role"]
            partial_messages = messages[:i+1]
            
            full_text = tokenizer.apply_chat_template(
                partial_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            
            if i == 0:
                new_ids = full_ids
            else:
                new_ids = full_ids[len(input_ids):]
            
            input_ids.extend(new_ids)
            
            if role == "user":
                labels.extend([-100] * len(new_ids))
            elif role == "assistant":
                labels.extend(new_ids)
        
        # 如果序列长度超过 max_length，丢弃该样本（返回空列表，后续过滤）
        if max_length is not None and len(input_ids) > max_length:
            input_ids_batch.append([])
            labels_batch.append([])
            continue
        
        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
    
    return {
        "input_ids": input_ids_batch,
        "labels": labels_batch
    }


def tokenize_vl_function_optimized(
    examples,
    tokenizer,
    max_length,
    fixed_image_token_count: int,
):
    """
    Optimized VL tokenization function for fixed resolution scenarios.
    
    **Core optimization**: When image resolution is fixed, <|image_pad|> token count is fixed.
    We don't need to load images and call processor for each sample, just:
    1. Use tokenizer to process pure text template
    2. Manually insert correct number of <|image_pad|> tokens
    
    Compared to original tokenize_vl_function:
    - No image file loading (saves I/O)
    - No processor image processing (saves computation)
    - Significant speed improvement (especially for large datasets)
    
    Args:
        examples: Batch of examples with 'messages' field
        tokenizer: Pre-loaded tokenizer (corresponding to VL model)
        max_length: Maximum sequence length
        fixed_image_token_count: Pre-computed image token count for fixed resolution
    
    Returns:
        Dictionary with tokenized data
    """
    image_token_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    
    input_ids_batch = []
    labels_batch = []
    has_image_batch = []
    
    for idx, messages in enumerate(examples["messages"]):
        modified_messages = []
        has_image = False
        
        for i, msg in enumerate(messages):
            if i == 0 and msg["role"] == "user":
                has_image = True
                modified_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": msg["content"]}
                    ]
                })
            else:
                modified_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })
        
        if has_image:
            text_only_messages = [{
                "role": msg["role"],
                "content": msg["content"] if isinstance(msg["content"], str) else 
                          "".join(c["text"] for c in msg["content"] if c.get("type") == "text")
            } for msg in messages]
            
            text_with_placeholder = tokenizer.apply_chat_template(
                modified_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            base_ids = tokenizer.encode(text_with_placeholder, add_special_tokens=False)
            
            input_ids = []
            i = 0
            while i < len(base_ids):
                if base_ids[i] == vision_start_id:
                    input_ids.append(vision_start_id)
                    i += 1
                    while i < len(base_ids) and base_ids[i] == image_token_id:
                        i += 1
                    input_ids.extend([image_token_id] * fixed_image_token_count)
                else:
                    input_ids.append(base_ids[i])
                    i += 1
            
            # 如果序列长度超过 max_length，丢弃该样本（返回空列表，后续过滤）
            if len(input_ids) > max_length:
                input_ids_batch.append([])
                labels_batch.append([])
                has_image_batch.append(False)
                continue
            
            has_image_batch.append(True)
        else:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            input_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
            # 如果序列长度超过 max_length，丢弃该样本（返回空列表，后续过滤）
            if max_length is not None and len(input_ids) > max_length:
                input_ids_batch.append([])
                labels_batch.append([])
                has_image_batch.append(False)
                continue
            has_image_batch.append(False)
        
        # Build labels
        labels = []
        current_ids = []
        
        text_messages = [{
            "role": msg["role"],
            "content": msg["content"] if isinstance(msg["content"], str) else
                      "".join(c.get("text", "") for c in msg["content"] if isinstance(c, dict) and c.get("type") == "text")
        } for msg in messages]
        
        for i, msg in enumerate(text_messages):
            role = msg["role"]
            partial = text_messages[:i+1]
            
            text = tokenizer.apply_chat_template(
                partial,
                tokenize=False,
                add_generation_prompt=False
            )
            full_ids = tokenizer.encode(text, add_special_tokens=False)
            
            if i == 0:
                new_ids = full_ids
            else:
                new_ids = full_ids[len(current_ids):]
            
            current_ids = full_ids
            
            if role == "user":
                labels.extend([-100] * len(new_ids))
            else:
                labels.extend(new_ids)
        
        # Adjust labels length to match input_ids
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids):
            labels = [-100] * (len(input_ids) - len(labels)) + labels
        
        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
    
    return {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "has_image": has_image_batch
    }


def tokenize_multiview_function(
    examples,
    tokenizer,
    max_length: int,
    fixed_image_token_count: Optional[int] = None,
):
    """
    Multi-view mode tokenization function.
    
    Handles samples with multiple images, inserting corresponding <|image_pad|> tokens for each.
    Supports two modes:
    1. Fixed resolution mode: Uses fixed_image_token_count for all images
    2. Original resolution mode: Uses examples["image_token_counts"] for each image
    
    Args:
        examples: Batch of examples with 'messages', 'num_images' and optionally 'image_token_counts'
        tokenizer: Qwen-VL tokenizer
        max_length: Maximum sequence length
        fixed_image_token_count: Per-image token count (optional, uses image_token_counts if None)
    
    Returns:
        Dictionary with tokenized data
    """
    image_token_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    
    use_dynamic_token_counts = fixed_image_token_count is None and "image_token_counts" in examples
    
    input_ids_batch = []
    labels_batch = []
    has_image_batch = []
    
    for idx, messages in enumerate(examples["messages"]):
        num_images = examples["num_images"][idx]
        
        if num_images > 0:
            if use_dynamic_token_counts:
                per_image_token_counts = examples["image_token_counts"][idx]
            else:
                per_image_token_counts = [fixed_image_token_count] * num_images
            
            modified_messages = []
            for i, msg in enumerate(messages):
                if i == 0 and msg["role"] == "user":
                    content = [{"type": "image"} for _ in range(num_images)]
                    content.append({"type": "text", "text": msg["content"]})
                    modified_messages.append({"role": "user", "content": content})
                else:
                    modified_messages.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
            
            text_with_placeholder = tokenizer.apply_chat_template(
                modified_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            base_ids = tokenizer.encode(text_with_placeholder, add_special_tokens=False)
            
            input_ids = []
            i = 0
            image_idx = 0
            while i < len(base_ids):
                if base_ids[i] == vision_start_id:
                    input_ids.append(vision_start_id)
                    i += 1
                    while i < len(base_ids) and base_ids[i] == image_token_id:
                        i += 1
                    token_count = per_image_token_counts[image_idx] if image_idx < len(per_image_token_counts) else per_image_token_counts[-1]
                    input_ids.extend([image_token_id] * token_count)
                    image_idx += 1
                else:
                    input_ids.append(base_ids[i])
                    i += 1
            
            # 如果序列长度超过 max_length，丢弃该样本（返回空列表，后续过滤）
            if len(input_ids) > max_length:
                input_ids_batch.append([])
                labels_batch.append([])
                has_image_batch.append(False)
                continue
            
            has_image_batch.append(True)
        else:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            input_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            # 如果序列长度超过 max_length，丢弃该样本（返回空列表，后续过滤）
            if max_length is not None and len(input_ids) > max_length:
                input_ids_batch.append([])
                labels_batch.append([])
                has_image_batch.append(False)
                continue
            has_image_batch.append(False)
        
        # Build labels
        labels = []
        current_ids = []
        text_messages = [{
            "role": msg["role"],
            "content": msg["content"] if isinstance(msg["content"], str) else
                      "".join(c.get("text", "") for c in msg["content"] if isinstance(c, dict) and c.get("type") == "text")
        } for msg in messages]
        
        for i, msg in enumerate(text_messages):
            role = msg["role"]
            partial = text_messages[:i+1]
            text = tokenizer.apply_chat_template(partial, tokenize=False, add_generation_prompt=False)
            full_ids = tokenizer.encode(text, add_special_tokens=False)
            
            new_ids = full_ids if i == 0 else full_ids[len(current_ids):]
            current_ids = full_ids
            
            if role == "user":
                labels.extend([-100] * len(new_ids))
            else:
                labels.extend(new_ids)
        
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids):
            labels = [-100] * (len(input_ids) - len(labels)) + labels
        
        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
    
    return {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "has_image": has_image_batch
    }


def tokenize_vl_function(
    examples,
    model_name,
    max_length,
    min_pixels=None,
    max_pixels=None,
    archive_base_path: Optional[str] = None,
    fixed_image_token_count: Optional[int] = None,
    tokenizer=None,
):
    """
    Tokenization function for Vision-Language models (e.g., Qwen-VL).
    
    This function handles both text and images, storing the processed
    data in a format compatible with VL model training.
    
    For Qwen2-VL/Qwen3-VL models, this function:
    1. Uses the full processor to tokenize text WITH image placeholders (<|image_pad|> tokens)
    2. The number of <|image_pad|> tokens depends on image resolution (determined by image_grid_thw)
    3. This ensures input_ids length matches what the model expects during forward pass
    
    NOTE: This function creates processor internally to support multiprocessing.
    Each subprocess will load its own processor to avoid pickle serialization issues.
    
    **Optimization mode**: When fixed_image_token_count is provided, uses optimized version
    without loading images. Suitable for fixed resolution scenarios.
    
    Args:
        examples: Batch of examples with 'messages' and optionally 'image' fields
        model_name: HuggingFace model name for loading processor
        max_length: Maximum sequence length
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing
        archive_base_path: Base path for image archives (not used in optimized mode)
        fixed_image_token_count: Pre-computed image token count (uses optimized mode if provided)
        tokenizer: Pre-loaded tokenizer (required for optimized mode)
    
    Returns:
        Dictionary with tokenized data including image features
    """
    # Use optimized version if fixed_image_token_count is provided
    if fixed_image_token_count is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when fixed_image_token_count is provided")
        return tokenize_vl_function_optimized(
            examples=examples,
            tokenizer=tokenizer,
            max_length=max_length,
            fixed_image_token_count=fixed_image_token_count,
        )
    
    # Create processor internally for multiprocessing support
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    input_ids_batch = []
    labels_batch = []
    has_image_batch = []
    
    has_image_refs = "image_ref" in examples
    
    for idx, messages in enumerate(examples["messages"]):
        image = None
        if archive_base_path and has_image_refs:
            image_ref = examples["image_ref"][idx]
            image = load_image_from_archive_ref(image_ref, archive_base_path)
        
        if image is not None:
            modified_messages = []
            for i, msg in enumerate(messages):
                if i == 0 and msg["role"] == "user":
                    modified_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": msg["content"]}
                        ]
                    })
                else:
                    modified_messages.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
            
            text = processor.apply_chat_template(
                modified_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            process_kwargs = {
                "text": [text],
                "images": [image],
                "return_tensors": None,
                "padding": False,
                "truncation": True,
                "max_length": max_length,
            }
            if min_pixels is not None:
                process_kwargs["min_pixels"] = min_pixels
            if max_pixels is not None:
                process_kwargs["max_pixels"] = max_pixels
                
            inputs = processor(**process_kwargs)
            input_ids = inputs["input_ids"][0]
            
            has_image_batch.append(True)
        else:
            text = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            input_ids = processor.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length
            )
            
            has_image_batch.append(False)
        
        # Build labels
        labels = []
        current_ids = []
        
        for i, msg in enumerate(messages):
            role = msg["role"]
            partial = messages[:i+1]
            
            text = processor.tokenizer.apply_chat_template(
                partial,
                tokenize=False,
                add_generation_prompt=False
            )
            full_ids = processor.tokenizer.encode(text, add_special_tokens=False)
            
            if i == 0:
                new_ids = full_ids
            else:
                new_ids = full_ids[len(current_ids):]
            
            current_ids = full_ids
            
            if role == "user":
                labels.extend([-100] * len(new_ids))
            else:
                labels.extend(new_ids)
        
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids):
            labels = [-100] * (len(input_ids) - len(labels)) + labels
        
        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
    
    result = {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "has_image": has_image_batch
    }
    
    return result
