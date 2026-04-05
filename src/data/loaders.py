"""
Data loading utilities for various input formats.
Supports JSON files, caption-JSON pairs, panorama-JSON pairs, and multi-view data.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("loaders")


def load_json_files(json_dir: str) -> List[Dict[str, Any]]:
    """
    Load all JSON files from a directory.
    
    Args:
        json_dir: Directory containing JSON files
    
    Returns:
        List of loaded JSON data
    """
    json_path = Path(json_dir)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    
    json_files = list(json_path.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    logger.info(f"Loading {len(json_files)} JSON files")
    
    data = []
    for json_file in tqdm(json_files, desc="Loading JSON files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                data.append(content)
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")
    
    logger.info(f"Successfully loaded {len(data)} files")
    return data


def load_caption_json_pairs(json_dir: str, caption_dir: str) -> List[Dict[str, Any]]:
    """
    Load paired caption and JSON files from directories.
    
    Args:
        json_dir: Directory containing JSON files
        caption_dir: Directory containing caption text files
    
    Returns:
        List of dictionaries with 'caption', 'json_data', and 'filename' keys
    """
    json_path = Path(json_dir)
    caption_path = Path(caption_dir)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    if not caption_path.exists():
        raise FileNotFoundError(f"Caption directory not found: {caption_dir}")
    
    caption_files = {f.stem: f for f in caption_path.glob("*.txt")}
    json_files = {f.stem: f for f in json_path.glob("*.json")}
    
    matched_stems = set(caption_files.keys()) & set(json_files.keys())
    
    if not matched_stems:
        raise ValueError(f"No matching caption-JSON pairs found between {json_dir} and {caption_dir}")
    
    logger.info(f"Found {len(matched_stems)} matching caption-JSON pairs")
    logger.info(f"Caption files: {len(caption_files)}, JSON files: {len(json_files)}")
    
    data = []
    for stem in tqdm(sorted(matched_stems), desc="Loading caption-JSON pairs"):
        try:
            with open(caption_files[stem], 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            with open(json_files[stem], 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            data.append({
                "caption": caption,
                "json_data": json_data,
                "filename": stem
            })
        except Exception as e:
            logger.error(f"Failed to load pair {stem}: {e}")
    
    logger.info(f"Successfully loaded {len(data)} pairs")
    return data


def load_panorama_json_pairs(json_dir: str, image_dir: str) -> List[Dict[str, Any]]:
    """
    Load paired panorama images and JSON files from directories.
    
    Image path structure: image_dir/{json_stem}/panorama/panorama_rgb.png
    
    Args:
        json_dir: Directory containing JSON files
        image_dir: Directory containing panorama image folders
    
    Returns:
        List of dictionaries with 'image_path', 'json_data', and 'filename' keys
    """
    json_path = Path(json_dir)
    image_path = Path(image_dir)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    json_files = {f.stem: f for f in json_path.glob("*.json")}
    
    data = []
    missing_images = 0
    
    for stem, json_file in tqdm(sorted(json_files.items()), desc="Loading panorama-JSON pairs"):
        panorama_path = image_path / stem / "panorama" / "panorama_rgb.png"
        
        if not panorama_path.exists():
            missing_images += 1
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            data.append({
                "image_path": str(panorama_path),
                "json_data": json_data,
                "filename": stem
            })
        except Exception as e:
            logger.error(f"Failed to load pair {stem}: {e}")
    
    if missing_images > 0:
        logger.warning(f"{missing_images} JSON files have no matching panorama image")
    
    logger.info(f"Found {len(data)} matching panorama-JSON pairs")
    logger.info(f"Total JSON files: {len(json_files)}, Matched pairs: {len(data)}")
    
    return data


def _load_multiview_json_pairs_generic(
    json_dir: str,
    image_dir: str,
    num_views_range: tuple = (4, 8),
    subdir: Optional[str] = "orbit",
    glob_pattern: str = "orbit_*.png",
    mode_label: str = "multi_view"
) -> List[Dict[str, Any]]:
    """Shared implementation for multi-view style loaders."""
    import random

    json_path = Path(json_dir)
    image_path = Path(image_dir)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    min_views, max_views = num_views_range
    json_files = {f.stem: f for f in json_path.glob("*.json")}

    data = []
    missing_images = 0
    insufficient_views = 0

    desc = f"Loading {mode_label} multi-view pairs"
    for stem, json_file in tqdm(sorted(json_files.items()), desc=desc):
        target_dir = image_path / stem
        if subdir:
            target_dir = target_dir / subdir

        if not target_dir.exists():
            missing_images += 1
            continue

        view_images = sorted(target_dir.glob(glob_pattern))

        if len(view_images) < min_views:
            insufficient_views += 1
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            num_views = random.randint(min_views, min(max_views, len(view_images)))
            selected_images = random.sample(view_images, num_views)
            random.shuffle(selected_images)

            data.append({
                "image_paths": [str(p) for p in selected_images],
                "json_data": json_data,
                "filename": stem
            })
        except Exception as e:
            logger.error(f"Failed to load pair {stem}: {e}")

    if missing_images > 0:
        logger.warning(f"[{mode_label}] {missing_images} JSON files have no matching image folder")
    if insufficient_views > 0:
        logger.warning(f"[{mode_label}] {insufficient_views} samples have fewer than {min_views} views")

    logger.info(f"[{mode_label}] Found {len(data)} matching multi-view-JSON pairs")
    logger.info(f"[{mode_label}] Views range: {min_views}-{max_views}")

    return data


def load_multiview_json_pairs(
    json_dir: str,
    image_dir: str,
    num_views_range: tuple = (4, 8)
) -> List[Dict[str, Any]]:
    """
    Load paired multi-view images and JSON files from directories.
    
    Image path structure: image_dir/{json_stem}/orbit/orbit_{xxx}.png
    
    Args:
        json_dir: Directory containing JSON files
        image_dir: Directory containing multi-view image folders
        num_views_range: Tuple of (min_views, max_views)
    
    Returns:
        List of dictionaries with 'image_paths', 'json_data', and 'filename' keys
    """
    return _load_multiview_json_pairs_generic(
        json_dir=json_dir,
        image_dir=image_dir,
        num_views_range=num_views_range,
        subdir="orbit",
        glob_pattern="orbit_*.png",
        mode_label="multi_view"
    )


def load_multiview2_json_pairs(
    json_dir: str,
    image_dir: str,
    num_views_range: tuple = (4, 8)
) -> List[Dict[str, Any]]:
    """
    Load multi-view data where images are directly under image_dir/{json_stem}/viewX.png.
    
    Args:
        json_dir: Directory containing JSON files
        image_dir: Directory containing multi-view image folders
        num_views_range: Tuple of (min_views, max_views)
    
    Returns:
        List of dictionaries with 'image_paths', 'json_data', and 'filename' keys
    """
    return _load_multiview_json_pairs_generic(
        json_dir=json_dir,
        image_dir=image_dir,
        num_views_range=num_views_range,
        subdir=None,
        glob_pattern="*.png",
        mode_label="multi_view2"
    )


def load_multiview3_json_pairs(
    json_dir: str,
    image_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load single front-view image and JSON files from directories.
    
    仅加载 image_dir/{json_stem}/front.png 一个固定视角。
    
    Image path structure: image_dir/{json_stem}/front.png
    
    Args:
        json_dir: Directory containing JSON files
        image_dir: Directory containing image folders (each with front.png)
    
    Returns:
        List of dictionaries with 'image_paths', 'json_data', and 'filename' keys
    """
    json_path = Path(json_dir)
    image_path = Path(image_dir)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    json_files = {f.stem: f for f in json_path.glob("*.json")}

    data = []
    missing_images = 0

    for stem, json_file in tqdm(sorted(json_files.items()), desc="Loading multi_view3 (front-only) pairs"):
        front_image = image_path / stem / "front.png"

        if not front_image.exists():
            missing_images += 1
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            data.append({
                "image_paths": [str(front_image)],
                "json_data": json_data,
                "filename": stem
            })
        except Exception as e:
            logger.error(f"Failed to load pair {stem}: {e}")

    if missing_images > 0:
        logger.warning(f"[multi_view3] {missing_images} JSON files have no matching front.png")

    logger.info(f"[multi_view3] Found {len(data)} matching front-view-JSON pairs")

    return data
