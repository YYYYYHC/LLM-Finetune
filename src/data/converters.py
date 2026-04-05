"""
Data format converters for transforming raw data into model-specific formats.
"""

import json
from typing import Dict, List, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("converters")


def convert_to_qwen_format(
    data: List[Dict[str, Any]],
    instruction_key: str = "instruction",
    output_key: str = "output",
    mode: str = "unconditional"
) -> List[Dict[str, Any]]:
    """
    Convert raw data to Qwen conversation format.
    
    Supports multiple generation modes:
    - "unconditional": Model generates complete scene from simple prompt (default)
    - "blueprint": Model generates objects based on blueprint condition
    - "caption": Model generates scene JSON based on text description
    - "panorama": Model generates scene JSON based on panorama image (VL model)
    - "multi_view" / "multi_view2": Model generates scene JSON based on multi-view images (VL model)
    - "contour": Model generates B-spline contour slices for a 3D shape

    Args:
        data: List of raw data dictionaries (scene data from YAML/JSON)
        instruction_key: Unused, kept for backward compatibility
        output_key: Unused, kept for backward compatibility
        mode: Generation mode ("unconditional", "blueprint", "caption", "panorama", "multi_view", "multi_view2", "contour")
    
    Returns:
        List of formatted conversations (with image_path for panorama mode)
    """
    formatted_data = []
    
    if mode == "caption":
        logger.info("Converting to Qwen format for caption-conditional generation (caption -> JSON)")
        
        for item in data:
            caption = item.get("caption")
            json_data = item.get("json_data")
            
            if not caption or not json_data:
                logger.warning(f"Item missing caption or json_data, skipping: {item.get('filename', 'unknown')}")
                continue
            
            json_output = json.dumps(json_data, ensure_ascii=False, separators=(',', ':'))
            
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Generate a 3D scene in JSON format based on the following description:\n{caption}"
                    },
                    {
                        "role": "assistant",
                        "content": json_output
                    }
                ]
            }
            formatted_data.append(conversation)
        
        logger.info(f"Created {len(formatted_data)} caption-conditional generation samples")
    
    elif mode == "blueprint":
        logger.info("Converting to Qwen format for blueprint-conditional generation (blueprint -> objects)")
        
        for item in data:
            blueprint = item.get("blueprint")
            if not blueprint:
                logger.warning("Item missing blueprint, skipping")
                continue
            
            objects = {k: v for k, v in item.items() if k.startswith("object_")}
            
            if not objects:
                logger.warning("Item has no objects, skipping")
                continue
            
            blueprint_str = json.dumps(blueprint, ensure_ascii=False, separators=(',', ':'))
            objects_str = json.dumps(objects, ensure_ascii=False, separators=(',', ':'))
            
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Generate 3D objects for the following room blueprint:\n{blueprint_str}"
                    },
                    {
                        "role": "assistant",
                        "content": objects_str
                    }
                ]
            }
            formatted_data.append(conversation)
        
        logger.info(f"Created {len(formatted_data)} blueprint-conditional generation samples")
        
    elif mode == "unconditional":
        logger.info("Converting to Qwen format for unconditional generation")
        
        for item in data:
            json_output = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
            
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Generate a 3D scene in JSON format:"
                    },
                    {
                        "role": "assistant",
                        "content": json_output
                    }
                ]
            }
            formatted_data.append(conversation)
        
        logger.info(f"Created {len(formatted_data)} unconditional generation samples")

    elif mode == "contour":
        logger.info("Converting to Qwen format for contour generation (B-spline contour slices)")

        for item in data:
            json_output = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Generate B-spline contour slices for a 3D shape in UDF contour token format:"
                    },
                    {
                        "role": "assistant",
                        "content": json_output
                    }
                ]
            }
            formatted_data.append(conversation)

        logger.info(f"Created {len(formatted_data)} contour generation samples")

    elif mode == "panorama":
        logger.info("Converting to Qwen-VL format for panorama-conditional generation (image -> JSON)")
        
        for item in data:
            image_path = item.get("image_path")
            json_data = item.get("json_data")
            
            if not image_path or not json_data:
                logger.warning(f"Item missing image_path or json_data, skipping: {item.get('filename', 'unknown')}")
                continue
            
            json_output = json.dumps(json_data, ensure_ascii=False, separators=(',', ':'))
            
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Based on this panorama image, generate a 3D scene in JSON format:"
                    },
                    {
                        "role": "assistant",
                        "content": json_output
                    }
                ],
                "image_path": image_path
            }
            formatted_data.append(conversation)
        
        logger.info(f"Created {len(formatted_data)} panorama-conditional generation samples")
    
    elif mode in ("multi_view", "multi_view2", "multi_view3"):
        logger.info("Converting to Qwen-VL format for multi-view-conditional generation (images -> JSON)")
        
        for item in data:
            image_paths = item.get("image_paths", [])
            json_data = item.get("json_data")
            
            if not image_paths or not json_data:
                logger.warning(f"Item missing image_paths or json_data, skipping: {item.get('filename', 'unknown')}")
                continue
            
            json_output = json.dumps(json_data, ensure_ascii=False, separators=(',', ':'))
            
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Based on these {len(image_paths)} multi-view images, generate a 3D scene in JSON format:"
                    },
                    {
                        "role": "assistant",
                        "content": json_output
                    }
                ],
                "image_paths": image_paths
            }
            formatted_data.append(conversation)
        
        logger.info(f"Created {len(formatted_data)} multi-view-conditional generation samples")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: 'unconditional', 'blueprint', 'caption', 'panorama', 'multi_view', 'multi_view2', 'multi_view3', 'contour'")
    
    return formatted_data
