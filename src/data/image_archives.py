"""
Image archive utilities for efficient storage and loading of images.
Supports tar-based archiving with indexing for fast random access.
"""

import io
import pickle
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("image_archives")

# Constants
IMAGE_ARCHIVE_INFO = "image_archives_info.json"
IMAGE_SHARD_TEMPLATE = "images_shard_{:04d}.tar"

# Global cache for tar file handles
_archive_handle_cache: Dict[str, tarfile.TarFile] = {}


def read_archive_info(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    """Read image archive info from dataset directory."""
    import json
    info_path = dataset_dir / IMAGE_ARCHIVE_INFO
    if not info_path.exists():
        return None
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to read {info_path}: {exc}")
        return None


def write_archive_info(dataset_dir: Path, base_key: str, archives_root: Path) -> None:
    """Write image archive info to dataset directory."""
    import json
    import os
    relpath = os.path.relpath(archives_root, dataset_dir)
    info_path = dataset_dir / IMAGE_ARCHIVE_INFO
    info = {
        "base_key": base_key,
        "archives_relpath": relpath,
        "format": "tar"
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def ensure_archives_root(output_path: Path, merge_mode: bool, existing_info: Optional[Dict[str, Any]]) -> Path:
    """Ensure archive root directory exists, optionally clearing it."""
    import shutil
    if existing_info and "archives_relpath" in existing_info:
        root = (output_path / existing_info["archives_relpath"]).resolve()
    else:
        root = output_path / "image_archives"
    if not merge_mode and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_and_encode_image(args) -> Optional[tuple]:
    """
    Load a single image, optionally resize, and encode to bytes.
    Used for multiprocessing to parallelize both I/O and encoding.
    
    Args:
        args: Tuple of (index, image_path, target_size, image_format, image_quality)
            - index: Original index of the image (for tracking)
            - image_path: Path to the image file, or None
            - target_size: Optional tuple (width, height) for resizing
            - image_format: 'png' or 'jpeg'
            - image_quality: Quality for jpeg (1-100), ignored for png
    
    Returns:
        Tuple of (index, image_bytes) or (index, None) if loading fails
    """
    idx, image_path, target_size, image_format, image_quality = args
    
    if not image_path:
        return (idx, None)
    try:
        img = Image.open(image_path).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        if image_format == "jpeg":
            img.save(buffer, format="JPEG", quality=image_quality)
        else:
            img.save(buffer, format="PNG")
        return (idx, buffer.getvalue())
    except Exception as e:
        print(f"Failed to load/encode image {image_path}: {e}")
        return (idx, None)


def load_and_encode_image_dynamic(args) -> Optional[tuple]:
    """
    Load image with dynamic resolution sampling and center crop.
    For multi-view mode with random resolution augmentation.
    
    Args:
        args: Tuple of (index, image_path, min_size, max_size, image_format, image_quality)
            - index: Original index of the image
            - image_path: Path to the image file
            - min_size: Tuple (min_width, min_height)
            - max_size: Tuple (max_width, max_height)
            - image_format: 'png' or 'jpeg'
            - image_quality: Quality for jpeg (1-100), ignored for png
    
    Returns:
        Tuple of (index, image_bytes, (actual_width, actual_height)) or (index, None, None)
    """
    import random
    idx, image_path, min_size, max_size, image_format, image_quality = args
    
    if not image_path:
        return (idx, None, None)
    try:
        img = Image.open(image_path).convert("RGB")
        
        # Sample target resolution independently for this view
        # 分辨率采样为32的倍数，以提高 token count 缓存命中率
        # 计算32倍数范围内的可选值数量
        min_w_steps = min_size[0] // 32
        max_w_steps = max_size[0] // 32
        min_h_steps = min_size[1] // 32
        max_h_steps = max_size[1] // 32
        target_w = random.randint(min_w_steps, max_w_steps) * 32
        target_h = random.randint(min_h_steps, max_h_steps) * 32
        
        # Resize image to max(target_w, target_h) x max(target_w, target_h)
        max_dim = max(target_w, target_h)
        img = img.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
        
        # Center crop to target size
        left = (max_dim - target_w) // 2
        top = (max_dim - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        img = img.crop((left, top, right, bottom))
        
        buffer = io.BytesIO()
        if image_format == "jpeg":
            img.save(buffer, format="JPEG", quality=image_quality)
        else:
            img.save(buffer, format="PNG")
        return (idx, buffer.getvalue(), (target_w, target_h))
    except Exception as e:
        print(f"Failed to load/encode image {image_path}: {e}")
        return (idx, None, None)


def get_image_resolution(image_path: str) -> Optional[tuple]:
    """
    Get the original resolution of an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        tuple: (width, height) or None if reading fails
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Failed to get resolution from {image_path}: {e}")
        return None


def load_single_image(args) -> Optional[Image.Image]:
    """
    Load a single image from path and optionally resize.
    
    Args:
        args: Tuple of (image_path, target_size) or just image_path
    
    Returns:
        PIL Image object or None if loading fails
    """
    if isinstance(args, tuple):
        image_path, target_size = args
    else:
        image_path = args
        target_size = None
    
    if not image_path:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None


def iter_images_encoded(image_paths: List[str], target_size: Optional[tuple], num_proc: int,
                        image_format: str = "png", image_quality: int = 85):
    """
    Parallel load images and encode to bytes.
    
    Optimizations:
    1. Image loading and encoding done in subprocesses
    2. Uses imap_unordered for better parallel efficiency
    3. Returns (index, bytes) tuples for correct index mapping
    
    Args:
        image_paths: List of image file paths
        target_size: Target size (width, height), None means no resize
        num_proc: Number of parallel processes
        image_format: 'png' or 'jpeg'
        image_quality: Quality for jpeg (1-100), ignored for png
    
    Yields:
        Tuple of (original_index, image_bytes or None)
    """
    if num_proc is None or num_proc <= 1:
        for idx, path in enumerate(image_paths):
            result = load_and_encode_image((idx, path, target_size, image_format, image_quality))
            yield result
    else:
        load_args = [(idx, path, target_size, image_format, image_quality) for idx, path in enumerate(image_paths)]
        with Pool(processes=num_proc) as pool:
            for result in pool.imap_unordered(load_and_encode_image, load_args):
                yield result


def iter_images_encoded_dynamic(
    image_paths: List[str], 
    min_size: tuple, 
    max_size: tuple, 
    num_proc: int,
    image_format: str = "png",
    image_quality: int = 85
):
    """
    Parallel load images with dynamic resolution sampling and center crop.
    
    Args:
        image_paths: List of image file paths
        min_size: Minimum resolution (width, height)
        max_size: Maximum resolution (width, height)
        num_proc: Number of parallel processes
        image_format: 'png' or 'jpeg'
        image_quality: Quality for jpeg (1-100), ignored for png
    
    Yields:
        Tuple of (original_index, image_bytes or None, (actual_width, actual_height) or None)
    """
    if num_proc is None or num_proc <= 1:
        for idx, path in enumerate(image_paths):
            result = load_and_encode_image_dynamic((idx, path, min_size, max_size, image_format, image_quality))
            yield result
    else:
        load_args = [(idx, path, min_size, max_size, image_format, image_quality) for idx, path in enumerate(image_paths)]
        with Pool(processes=num_proc) as pool:
            for result in pool.imap_unordered(load_and_encode_image_dynamic, load_args):
                yield result


def create_image_archives(
    image_paths: List[str],
    archives_root: Path,
    target_size: Optional[tuple],
    num_proc: int,
    shard_size: int = 1000,
    subdir: Optional[str] = None,
    image_format: str = "png",
    image_quality: int = 85
) -> List[Optional[str]]:
    """
    Create image archives (tar format).
    
    Optimized version:
    1. Image loading and encoding done in parallel subprocesses
    2. Uses imap_unordered for better efficiency
    3. Builds complete tar data in memory, then writes to disk at once
    
    Args:
        image_paths: List of image file paths
        archives_root: Archive root directory
        target_size: Target size (width, height)
        num_proc: Number of parallel processes
        shard_size: Max images per tar file
        subdir: Subdirectory name (optional)
        image_format: 'png' or 'jpeg'
        image_quality: Quality for jpeg (1-100), ignored for png
    
    Returns:
        List of reference strings in format "tar_path::member_name"
    """
    if not image_paths:
        return []
    shard_size = max(1, shard_size)
    archives_dir = archives_root if subdir is None else archives_root / subdir
    archives_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = sorted(archives_dir.glob("images_shard_*.tar"))
    next_shard_idx = len(existing_shards)
    
    references: List[Optional[str]] = [None] * len(image_paths)
    pending_entries: List[tuple] = []

    def _flush_pending_to_memory_tar():
        """Build complete tar in memory and write to disk at once."""
        nonlocal pending_entries, next_shard_idx
        if not pending_entries:
            return
        
        tar_path = archives_dir / IMAGE_SHARD_TEMPLATE.format(next_shard_idx)
        next_shard_idx += 1
        rel_tar = tar_path.relative_to(archives_root)
        
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for orig_idx, member_name, data in pending_entries:
                tar_info = tarfile.TarInfo(name=member_name)
                tar_info.size = len(data)
                tar.addfile(tar_info, io.BytesIO(data))
                references[orig_idx] = f"{rel_tar.as_posix()}::{member_name}"
        
        tar_buffer.seek(0)
        with open(tar_path, "wb") as f:
            f.write(tar_buffer.getvalue())
        
        # Create index file for fast random access
        idx_path = tar_path.with_suffix(".tar.idx")
        index_data = {}
        with tarfile.open(tar_path, "r") as tar_read:
            for member in tar_read.getmembers():
                index_data[member.name] = (member.offset_data, member.size)
        with open(idx_path, "wb") as f:
            pickle.dump(index_data, f)
        
        pending_entries = []

    iterator = iter_images_encoded(image_paths, target_size, num_proc, image_format, image_quality)
    desc = "Archiving images" if subdir is None else f"Archiving images ({subdir})"
    ext = "jpg" if image_format == "jpeg" else "png"
    
    for orig_idx, data in tqdm(iterator, total=len(image_paths), desc=desc):
        if data is None:
            continue
        
        member_name = f"{Path(image_paths[orig_idx]).stem}_{orig_idx:08d}.{ext}"
        pending_entries.append((orig_idx, member_name, data))
        
        if len(pending_entries) >= shard_size:
            _flush_pending_to_memory_tar()
    
    _flush_pending_to_memory_tar()
    
    return references


def create_image_archives_dynamic(
    image_paths: List[str],
    archives_root: Path,
    min_size: tuple,
    max_size: tuple,
    num_proc: int,
    shard_size: int = 1000,
    subdir: Optional[str] = None,
    image_format: str = "png",
    image_quality: int = 85
) -> Tuple[List[Optional[str]], List[Optional[Tuple[int, int]]]]:
    """
    Create image archives with dynamic resolution sampling and center crop.
    Each image is independently sampled for resolution within the given range.
    
    Args:
        image_paths: List of image file paths
        archives_root: Archive root directory
        min_size: Minimum resolution (width, height)
        max_size: Maximum resolution (width, height)
        num_proc: Number of parallel processes
        shard_size: Max images per tar file
        subdir: Subdirectory name (optional)
        image_format: 'png' or 'jpeg'
        image_quality: Quality for jpeg (1-100), ignored for png
    
    Returns:
        Tuple of:
        - List of reference strings in format "tar_path::member_name"
        - List of actual resolutions (width, height) for each image
    """
    if not image_paths:
        return [], []
    shard_size = max(1, shard_size)
    archives_dir = archives_root if subdir is None else archives_root / subdir
    archives_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = sorted(archives_dir.glob("images_shard_*.tar"))
    next_shard_idx = len(existing_shards)
    
    references: List[Optional[str]] = [None] * len(image_paths)
    resolutions: List[Optional[Tuple[int, int]]] = [None] * len(image_paths)
    pending_entries: List[tuple] = []

    def _flush_pending_to_memory_tar():
        """Build complete tar in memory and write to disk at once."""
        nonlocal pending_entries, next_shard_idx
        if not pending_entries:
            return
        
        tar_path = archives_dir / IMAGE_SHARD_TEMPLATE.format(next_shard_idx)
        next_shard_idx += 1
        rel_tar = tar_path.relative_to(archives_root)
        
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for orig_idx, member_name, data, resolution in pending_entries:
                tar_info = tarfile.TarInfo(name=member_name)
                tar_info.size = len(data)
                tar.addfile(tar_info, io.BytesIO(data))
                references[orig_idx] = f"{rel_tar.as_posix()}::{member_name}"
                resolutions[orig_idx] = resolution
        
        tar_buffer.seek(0)
        with open(tar_path, "wb") as f:
            f.write(tar_buffer.getvalue())
        
        # Create index file for fast random access
        idx_path = tar_path.with_suffix(".tar.idx")
        index_data = {}
        with tarfile.open(tar_path, "r") as tar_read:
            for member in tar_read.getmembers():
                index_data[member.name] = (member.offset_data, member.size)
        with open(idx_path, "wb") as f:
            pickle.dump(index_data, f)
        
        pending_entries = []

    iterator = iter_images_encoded_dynamic(image_paths, min_size, max_size, num_proc, image_format, image_quality)
    desc = "Archiving images (dynamic)" if subdir is None else f"Archiving images (dynamic, {subdir})"
    ext = "jpg" if image_format == "jpeg" else "png"
    
    for orig_idx, data, resolution in tqdm(iterator, total=len(image_paths), desc=desc):
        if data is None:
            continue
        
        member_name = f"{Path(image_paths[orig_idx]).stem}_{orig_idx:08d}.{ext}"
        pending_entries.append((orig_idx, member_name, data, resolution))
        
        if len(pending_entries) >= shard_size:
            _flush_pending_to_memory_tar()
    
    _flush_pending_to_memory_tar()
    
    return references, resolutions


def load_image_from_archive_ref(image_ref: Optional[str], archive_base_path: Optional[str]) -> Optional[Image.Image]:
    """
    Load an image from archive reference.
    
    Args:
        image_ref: Reference string in format "tar_path::member_name"
        archive_base_path: Base path for archives
    
    Returns:
        PIL Image or None if loading fails
    """
    if not image_ref or not archive_base_path:
        return None
    try:
        archive_rel, member_name = image_ref.split("::", 1)
    except ValueError:
        return None
    archive_path = (Path(archive_base_path) / archive_rel).resolve()
    cache_key = str(archive_path)
    tar = _archive_handle_cache.get(cache_key)
    if tar is None:
        if not archive_path.exists():
            logger.warning(f"Archive not found: {archive_path}")
            return None
        tar = tarfile.open(archive_path, "r")
        _archive_handle_cache[cache_key] = tar
    member = tar.extractfile(member_name)
    if member is None:
        logger.warning(f"Image member {member_name} missing in {archive_path}")
        return None
    data = member.read()
    member.close()
    return Image.open(io.BytesIO(data)).convert("RGB")


def get_image_resolutions_batch(image_paths: List[str], num_proc: int = 1) -> List[Optional[tuple]]:
    """
    Batch get image resolutions.
    
    Args:
        image_paths: List of image file paths
        num_proc: Number of parallel processes
    
    Returns:
        List of (width, height) tuples, None for failed reads
    """
    if num_proc is None or num_proc <= 1:
        return [get_image_resolution(p) for p in image_paths]
    else:
        with Pool(processes=num_proc) as pool:
            return pool.map(get_image_resolution, image_paths)
