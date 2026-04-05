"""
Training logic for fine-tuning language models.
Supports both text-only and vision-language models.
"""

import os
import math
import signal
import json
import tarfile
import pickle
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    get_scheduler,
    set_seed
)
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image
import io

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger
from src.models.model_factory import create_model, load_tokenizer, load_processor, is_vision_language_model

logger = setup_logger("trainer")


def compute_vl_position_ids(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    image_token_id: int,
    vision_start_token_id: int,
    sequence_lengths: list = None,
) -> torch.Tensor:
    """
    Compute 3D position_ids for Qwen3-VL models with M-RoPE.
    
    This function replicates the logic from Qwen3VLModel.get_rope_index() to generate
    position_ids for packing support. The key difference is that it handles sub-sequence
    boundaries for packing mode.
    
    Reference: transformers/models/qwen3_vl/modeling_qwen3_vl.py :: get_rope_index()
    
    Args:
        input_ids: Token IDs tensor of shape (batch_size, seq_len)
        image_grid_thw: Image grid info tensor of shape (num_images, 3) where 3 = (t, h, w)
                       t is always 1 for images (timestamps used for temporal encoding)
                       h, w are the raw grid dimensions (before spatial_merge_size division)
        spatial_merge_size: The spatial merge size from vision config (typically 2)
                           LLM grid dimensions are h//spatial_merge_size and w//spatial_merge_size
        image_token_id: Token ID for image placeholder (from config.image_token_id)
        vision_start_token_id: Token ID for <|vision_start|> (from config.vision_start_token_id)
        sequence_lengths: For packing, list of sub-sequence lengths per batch item
                         If provided, position_ids will reset at sub-sequence boundaries
    
    Returns:
        position_ids: Tensor of shape (3, batch_size, seq_len) for M-RoPE
                     Dimension 0 has 3 components: (temporal, height, width)
                     This matches Qwen3-VL's expected input format.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Initialize position_ids with zeros
    # Shape: (3, batch_size, seq_len) - Qwen3-VL expects this format
    position_ids = torch.ones((3, batch_size, seq_len), dtype=input_ids.dtype, device=device)
    
    # Process each batch item
    global_image_idx = 0  # Global image index across batch
    
    for batch_idx in range(batch_size):
        if sequence_lengths is not None and sequence_lengths[batch_idx]:
            # Packing mode: handle multiple sub-sequences
            sub_seq_lengths = sequence_lengths[batch_idx]
            offset = 0
            
            for sub_len in sub_seq_lengths:
                # Process this sub-sequence
                sub_seq = input_ids[batch_idx, offset:offset + sub_len]
                
                # Count images in this sub-sequence
                vision_start_indices = torch.argwhere(sub_seq == vision_start_token_id).squeeze(1)
                if vision_start_indices.numel() > 0:
                    vision_tokens = sub_seq[vision_start_indices + 1]
                    num_images = (vision_tokens == image_token_id).sum().item()
                else:
                    num_images = 0
                
                # Extract image_grid_thw for this sub-sequence
                sub_image_grid_thw = image_grid_thw[global_image_idx:global_image_idx + num_images] if num_images > 0 else None
                
                # Compute position_ids for this sub-sequence
                sub_position_ids = _compute_single_sequence_position_ids_qwen3vl(
                    sub_seq,
                    sub_image_grid_thw,
                    spatial_merge_size,
                    image_token_id,
                    vision_start_token_id,
                )
                
                # Assign to position_ids: sub_position_ids is (3, sub_len)
                position_ids[:, batch_idx, offset:offset + sub_len] = sub_position_ids
                
                offset += sub_len
                global_image_idx += num_images
        else:
            # Non-packing mode: single sequence
            seq = input_ids[batch_idx]
            
            # Count images in this sequence
            vision_start_indices = torch.argwhere(seq == vision_start_token_id).squeeze(1)
            if vision_start_indices.numel() > 0:
                vision_tokens = seq[vision_start_indices + 1]
                num_images = (vision_tokens == image_token_id).sum().item()
            else:
                num_images = 0
            
            # Extract image_grid_thw for this sequence
            seq_image_grid_thw = image_grid_thw[global_image_idx:global_image_idx + num_images] if num_images > 0 else None
            
            single_pos_ids = _compute_single_sequence_position_ids_qwen3vl(
                seq,
                seq_image_grid_thw,
                spatial_merge_size,
                image_token_id,
                vision_start_token_id,
            )
            
            # single_pos_ids shape: (3, seq_len)
            position_ids[:, batch_idx, :] = single_pos_ids
            global_image_idx += num_images
    
    return position_ids


def build_varlen_position_ids(
    sequence_lengths: list,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate 1D position_ids that reset at every packed sub-sequence."""
    batch_size = len(sequence_lengths)
    packed_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for batch_idx, sub_sequences in enumerate(sequence_lengths):
        offset = 0
        for sub_len in sub_sequences:
            span = min(sub_len, max_len - offset)
            if span <= 0:
                break
            packed_position_ids[batch_idx, offset:offset + span] = torch.arange(span, device=device)
            offset += span

        if offset < max_len:
            pad_span = max_len - offset
            packed_position_ids[batch_idx, offset:] = torch.arange(pad_span, device=device)

    return packed_position_ids


def _compute_single_sequence_position_ids_qwen3vl(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    image_token_id: int,
    vision_start_token_id: int,
) -> torch.Tensor:
    """
    Compute 3D position_ids for a single sequence following Qwen3-VL's get_rope_index logic.
    
    Reference: Qwen3VLModel.get_rope_index() in modeling_qwen3_vl.py (lines 916-1033)
    
    Key differences from Qwen2-VL:
    - Uses spatial_merge_size to determine LLM grid dimensions
    - t dimension is always 0 for image tokens (timestamps encode temporal info)
    - Text tokens have same value for all 3 dimensions
    
    Args:
        input_ids: 1D tensor of token IDs for a single sequence
        image_grid_thw: Tensor of shape (num_images, 3) with (t, h, w) per image, or None
        spatial_merge_size: Vision config's spatial merge size
        image_token_id: Token ID for image placeholder
        vision_start_token_id: Token ID for <|vision_start|>
    
    Returns:
        position_ids: Tensor of shape (3, seq_len)
    """
    seq_len = input_ids.shape[0]
    device = input_ids.device
    
    input_tokens = input_ids.tolist()
    llm_pos_ids_list = []
    
    st = 0
    image_index = 0
    
    # Count total images
    vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
    if vision_start_indices.numel() > 0:
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum().item()
    else:
        image_nums = 0
    
    remain_images = image_nums
    
    # Process each image region
    for _ in range(image_nums):
        if image_token_id in input_tokens and remain_images > 0:
            ed_image = input_tokens.index(image_token_id, st)
        else:
            ed_image = len(input_tokens) + 1
        
        # Get grid dimensions for this image
        t, h, w = (
            image_grid_thw[image_index][0],
            image_grid_thw[image_index][1],
            image_grid_thw[image_index][2],
        )
        image_index += 1
        remain_images -= 1
        ed = ed_image
        
        # LLM grid dimensions (after spatial merge)
        llm_grid_t, llm_grid_h, llm_grid_w = (
            t.item(),
            h.item() // spatial_merge_size,
            w.item() // spatial_merge_size,
        )
        
        # Text tokens before this image
        text_len = ed - st
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx)
        
        # Image tokens position_ids
        # t_index is always 0 because llm_grid_t is always 1 (timestamps encode temporal)
        t_index = torch.arange(llm_grid_t, device=device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h, device=device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w, device=device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
        
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w
    
    # Remaining text tokens after all images
    if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx)
    
    # Concatenate all position_ids
    if llm_pos_ids_list:
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    else:
        # No tokens (shouldn't happen, but handle gracefully)
        llm_positions = torch.zeros((3, seq_len), dtype=torch.long, device=device)
    
    return llm_positions


def parse_batch_indices(batch_spec):
    """
    Parse batch indices from various formats.
    
    Supports:
    - Single integer: 0
    - List of integers: [0, 1, 2]
    - String range: "1-45"
    - List with mixed formats: [0, "5-10", 15]
    
    Args:
        batch_spec: Integer, string, or list containing batch specifications
    
    Returns:
        List of integers representing batch indices
    
    Examples:
        >>> parse_batch_indices(5)
        [5]
        >>> parse_batch_indices("1-5")
        [1, 2, 3, 4, 5]
        >>> parse_batch_indices([0, "5-10", 15])
        [0, 5, 6, 7, 8, 9, 10, 15]
    """
    result = []
    
    # Handle single integer
    if isinstance(batch_spec, int):
        return [batch_spec]
    
    # Handle string range like "1-45"
    if isinstance(batch_spec, str):
        if "-" in batch_spec:
            parts = batch_spec.split("-")
            if len(parts) == 2:
                start, end = int(parts[0].strip()), int(parts[1].strip())
                return list(range(start, end + 1))
        # Single number as string
        return [int(batch_spec)]
    
    # Handle list (can contain integers, strings, or ranges)
    if isinstance(batch_spec, list):
        for item in batch_spec:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, str):
                if "-" in item:
                    # Parse range
                    parts = item.split("-")
                    if len(parts) == 2:
                        start, end = int(parts[0].strip()), int(parts[1].strip())
                        result.extend(range(start, end + 1))
                else:
                    # Single number as string
                    result.append(int(item))
            else:
                raise ValueError(f"Unsupported batch specification type: {type(item)}")
        return result
    
    raise ValueError(f"Unsupported batch specification format: {type(batch_spec)}")


class Trainer:
    """
    Trainer class for fine-tuning language models.
    Supports both text-only and vision-language models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        
        # Set seed for reproducibility
        seed = config.get("seed", 42)
        set_seed(seed)
        accelerate_set_seed(seed)
        
        # Initialize accelerator
        # Explicitly pass gradient_accumulation_steps to avoid 'auto' conversion error
        # when using DeepSpeed config with 'auto' values
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        
        # Set up tensorboard log directory
        # By default, accelerate creates: project_dir/project_name/timestamp/
        # We want logs directly in output_dir to avoid nested subdirectories
        output_dir = config.get("output_dir", "./outputs")
        log_with = config.get("log_with", "tensorboard")
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=log_with,
            project_dir=output_dir
        )
        
        # Initialize tracker for logging
        if self.accelerator.is_main_process:
            experiment_config = {
                "learning_rate": config.get("learning_rate", 2e-5),
                "num_epochs": config.get("num_epochs", 3),
                "batch_size": config.get("per_device_train_batch_size", 1),
                "model_name": config.get("model_name", ""),
                "training_mode": config.get("training_mode", "full")
            }
            # Use output_dir name as project_name to avoid extra nesting
            project_name = Path(output_dir).name
            self.accelerator.init_trackers(
                project_name=project_name,
                config=experiment_config
            )
        
        # Setup logging
        if self.accelerator.is_main_process:
            logger.info("Initializing trainer")
            logger.info(f"Config: {config}")
        
        # Check if this is a VL model
        self.is_vl_model = config.get("is_vl_model", None)
        if self.is_vl_model is None:
            self.is_vl_model = is_vision_language_model(config["model_name"])
        
        if self.is_vl_model:
            if self.accelerator.is_main_process:
                logger.info("Detected Vision-Language model, loading processor...")
            # Load processor for VL models
            self.processor = load_processor(
                config["model_name"],
                padding_side=config.get("padding_side", "right")
            )
            self.tokenizer = self.processor.tokenizer
        else:
            # Load tokenizer for text-only models
            self.tokenizer = load_tokenizer(
                config["model_name"],
                padding_side=config.get("padding_side", "right")
            )
            self.processor = None
            self.image_base_mapping: Dict[str, Dict[str, Any]] = {}
            self._image_archive_cache: Dict[str, tarfile.TarFile] = {}
        
        # Load model
        self.model = self._load_model()
        
        # Load dataset
        self.train_dataset, self.eval_dataset = self._load_datasets()
        
        # Create dataloaders
        self.train_dataloader = self._create_dataloader(
            self.train_dataset,
            batch_size=config.get("per_device_train_batch_size", 1),
            shuffle=True
        )
        
        self.eval_dataloader = None
        if self.eval_dataset is not None:
            self.eval_dataloader = self._create_dataloader(
                self.eval_dataset,
                batch_size=config.get("per_device_eval_batch_size", 1),
                shuffle=False
            )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.lr_scheduler = self._create_scheduler()
        
        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler
        )
        
        if self.eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Get max_length for dynamic truncation in collate_fn
        self.max_length = config.get("max_length", None)
        
        # Get pad_to_max_length flag for testing memory peak
        self.pad_to_max_length = config.get("pad_to_max_length", False)
        
        if self.pad_to_max_length:
            if self.max_length is None:
                raise ValueError(
                    "pad_to_max_length is enabled but max_length is not set. "
                    "Please specify max_length in the config."
                )
            if self.accelerator.is_main_process:
                logger.warning(
                    f"pad_to_max_length is enabled. All sequences will be padded to {self.max_length}. "
                    "This will significantly increase memory usage and is intended for testing memory peak. "
                    "Disable this for actual training to use dynamic padding."
                )
        
        # Save configuration to output directory for record keeping
        if self.accelerator.is_main_process:
            self._save_config()
        self._log_lora_target_modules()
        self._log_all_model_layers()
        
        # Resume from checkpoint if specified (恢复完整训练状态)
        resume_from_checkpoint = config.get("resume_from_checkpoint", None)
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Initialize from checkpoint if specified (仅加载权重，不恢复训练状态)
        init_from_checkpoint = config.get("init_from_checkpoint", None)
        if init_from_checkpoint:
            self.load_weights_only(init_from_checkpoint)
        
        # 设置信号处理器，支持 Ctrl+C 优雅退出并保存模型
        self._interrupted = False
        self._saving = False  # 标记是否正在保存
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """
        处理 Ctrl+C (SIGINT) 信号，设置中断标志以便优雅退出。
        第一次按 Ctrl+C 会在当前 step 结束后保存并退出。
        第二次按 Ctrl+C 会强制退出（即使正在保存也会退出）。
        """
        if self._interrupted:
            # 第二次 Ctrl+C，强制退出（无论是否正在保存）
            if self.accelerator.is_main_process:
                if self._saving:
                    logger.warning("收到第二次中断信号，跳过保存，强制退出...")
                else:
                    logger.warning("收到第二次中断信号，强制退出...")
            # 恢复原始信号处理器并重新发送信号
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            os.kill(os.getpid(), signal.SIGINT)
        else:
            self._interrupted = True
            if self.accelerator.is_main_process:
                logger.warning("\n收到中断信号 (Ctrl+C)，将在当前 step 结束后保存模型并退出...")
                logger.warning("再次按 Ctrl+C 可强制退出（不保存）")
    
    def _load_model(self):
        """Load model based on configuration."""
        training_mode = self.config.get("training_mode", "full")
        
        model_kwargs = {
            "torch_dtype": self.config.get("torch_dtype", "bfloat16"),
            "use_flash_attention": self.config.get("use_flash_attention", True),
            "is_vl_model": self.is_vl_model
        }
        
        if training_mode == "lora":
            model_kwargs["lora_config"] = self.config.get("lora_config", {})
            model_kwargs["load_in_4bit"] = self.config.get("load_in_4bit", False)
            model_kwargs["load_in_8bit"] = self.config.get("load_in_8bit", False)
        elif training_mode == "full" and self.is_vl_model:
            # 支持冻结 Vision Encoder 和 Merger（仅对 VL 模型全量微调有效）
            model_kwargs["freeze_vision_encoder"] = self.config.get("freeze_vision_encoder", True)
            model_kwargs["freeze_vision_merger"] = self.config.get("freeze_vision_merger", True)
        
        return create_model(
            self.config["model_name"],
            training_mode=training_mode,
            **model_kwargs
        )
    
    def _load_datasets(self):
        """Load datasets from disk or from batch directories."""
        from pathlib import Path
        from datasets import concatenate_datasets
        self.image_base_mapping = {}
        self._image_archive_cache = {}
        
        # Check if using batch directories mode
        if "batch_dirs" in self.config:
            return self._load_from_batches()
        
        # Normal mode: load from single directory
        dataset_path = self.config["dataset_path"]
        
        if self.accelerator.is_main_process:
            logger.info(f"Loading dataset from {dataset_path}")
        
        dataset = load_from_disk(dataset_path)
        self._register_image_archives(Path(dataset_path))
        
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("test", None)
        
        if self.accelerator.is_main_process:
            logger.info(f"Train samples: {len(train_dataset)}")
            if eval_dataset:
                logger.info(f"Eval samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def _load_from_batches(self):
        """Load datasets from multiple batch directories.
        
        Supports two config formats:
        1. Single directory (legacy):
           batch_dirs:
             base_dir: "/path/to/dir"
             train_batches: "1-20"
             eval_batches: [0]
        
        2. Multiple directories:
           batch_dirs:
             - base_dir: "/path/to/dir1"
               train_batches: "1-20"
               eval_batches: [0]
             - base_dir: "/path/to/dir2"
               train_batches: "1-15"
               eval_batches: [0]
        """
        from pathlib import Path
        from datasets import concatenate_datasets, Dataset
        
        batch_config = self.config["batch_dirs"]
        
        # Normalize to list format for unified processing
        if isinstance(batch_config, dict):
            # Legacy single directory format
            batch_configs = [batch_config]
        else:
            # New multiple directories format (list)
            batch_configs = batch_config
        
        all_train_datasets = []
        all_eval_datasets = []
        
        for idx, config in enumerate(batch_configs):
            base_dir = Path(config["base_dir"])
            train_batches_spec = config.get("train_batches", [])
            eval_batches_spec = config.get("eval_batches", [])
            
            # Parse batch specifications (supports ranges like "1-45")
            train_batches = parse_batch_indices(train_batches_spec)
            eval_batches = parse_batch_indices(eval_batches_spec) if eval_batches_spec else []
            
            if self.accelerator.is_main_process:
                logger.info(f"[Dir {idx+1}/{len(batch_configs)}] Loading from: {base_dir}")
                logger.info(f"  Train batches: {train_batches_spec} -> {train_batches}")
                if eval_batches:
                    logger.info(f"  Eval batches: {eval_batches_spec} -> {eval_batches}")
            
            # Load training batches from this directory
            for batch_idx in train_batches:
                batch_dir = base_dir / f"batch_{batch_idx:04d}"
                if self.accelerator.is_main_process:
                    logger.info(f"    Loading train batch {batch_idx}")
                dataset = Dataset.load_from_disk(str(batch_dir))
                self._register_image_archives(batch_dir)
                all_train_datasets.append(dataset)
            
            # Load eval batches from this directory
            for batch_idx in eval_batches:
                batch_dir = base_dir / f"batch_{batch_idx:04d}"
                if self.accelerator.is_main_process:
                    logger.info(f"    Loading eval batch {batch_idx}")
                dataset = Dataset.load_from_disk(str(batch_dir))
                self._register_image_archives(batch_dir)
                all_eval_datasets.append(dataset)
        
        # Concatenate all datasets
        train_dataset = concatenate_datasets(all_train_datasets) if len(all_train_datasets) > 1 else all_train_datasets[0]
        eval_dataset = concatenate_datasets(all_eval_datasets) if all_eval_datasets else None
        
        if self.accelerator.is_main_process:
            logger.info(f"Total train samples: {len(train_dataset)}")
            if eval_dataset:
                logger.info(f"Total eval samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset

    def _register_image_archives(self, dataset_dir: Path) -> None:
        info_path = Path(dataset_dir) / "image_archives_info.json"
        if not info_path.exists():
            return
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to read image archive info from {info_path}: {exc}")
            return
        base_key = info.get("base_key")
        relpath = info.get("archives_relpath")
        if not base_key or not relpath:
            return
        archives_path = (Path(dataset_dir) / relpath).resolve()
        self.image_base_mapping[base_key] = {
            "path": archives_path,
            "format": info.get("format", "tar")
        }

    def _load_image_from_archive(self, image_ref: Optional[str], base_key: Optional[str]) -> Optional[Image.Image]:
        """使用索引文件实现O(1)随机访问tar中的图像。"""
        if not image_ref or not base_key:
            return None
        base_entry = self.image_base_mapping.get(base_key)
        if not base_entry:
            # 直接报错：如果 base_key 未注册，说明 image_archives_info.json 缺失
            raise RuntimeError(
                f"图像加载失败：base_key '{base_key}' 未在 image_base_mapping 中注册！\n"
                f"请检查数据集目录是否包含 image_archives_info.json 文件。\n"
                f"当前已注册的 base_keys: {list(self.image_base_mapping.keys())}\n"
                f"可使用 scripts/fix_multiview_archives_info.py 脚本修复缺失的 image_archives_info.json 文件。"
            )
        try:
            archive_rel, member_name = image_ref.split("::", 1)
        except ValueError:
            return None
        archive_path = (base_entry["path"] / archive_rel).resolve()
        cache_key = str(archive_path)
        
        # 初始化索引和文件句柄缓存（如果不存在）
        if not hasattr(self, '_tar_index_cache'):
            self._tar_index_cache: Dict[str, Dict[str, tuple]] = {}
        if not hasattr(self, '_tar_file_cache'):
            self._tar_file_cache: Dict[str, Any] = {}
        
        # 加载或获取索引
        index = self._tar_index_cache.get(cache_key)
        if index is None:
            idx_path = archive_path.with_suffix(".tar.idx")
            if idx_path.exists():
                with open(idx_path, "rb") as f:
                    index = pickle.load(f)
                self._tar_index_cache[cache_key] = index
            else:
                index = None  # 无索引，回退到传统方式
        
        # 使用索引进行O(1)访问
        if index is not None and member_name in index:
            offset, size = index[member_name]
            fp = self._tar_file_cache.get(cache_key)
            if fp is None:
                if not archive_path.exists():
                    logger.warning(f"Archive not found: {archive_path}")
                    return None
                fp = open(archive_path, "rb")
                self._tar_file_cache[cache_key] = fp
            fp.seek(offset)
            data = fp.read(size)
        else:
            # 回退：无索引时使用tarfile（兼容旧数据）
            tar = self._image_archive_cache.get(cache_key)
            if tar is None:
                if not archive_path.exists():
                    logger.warning(f"Archive not found: {archive_path}")
                    return None
                tar = tarfile.open(archive_path, "r")
                self._image_archive_cache[cache_key] = tar
            member = tar.extractfile(member_name)
            if member is None:
                logger.warning(f"Image {member_name} missing in {archive_path}")
                return None
            data = member.read()
            member.close()
        
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            logger.warning(f"Failed to decode image {member_name} from {archive_path}: {exc}")
            return None
    
    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool):
        """Create dataloader for dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=self.config.get("dataloader_num_workers", 0),
            pin_memory=True
        )
    
    def _collate_fn(self, batch):
        """
        Collate function for dataloader with dynamic padding and truncation support.
        
        This function dynamically pads sequences to the longest sequence in the batch,
        which is much more efficient than padding all sequences to max_length.
        
        If max_length is specified in config, sequences will be truncated
        to this length during training.
        
        If pad_to_max_length is True, all sequences will be padded to max_length
        instead of the longest sequence in the batch. This is useful for testing
        memory peak to determine optimal batch size.
        
        For packed sequences (with 'sequence_lengths' field), generates position_ids
        to isolate attention between sub-sequences. Uses Flash Attention 2's
        flash_attn_varlen_func for efficient attention computation with packing.
        
        For VL models (Qwen2-VL/Qwen3-VL), generates 3D position_ids in M-RoPE format:
        (batch_size, seq_len, 3) where 3 = (temporal, height, width).
        
        For VL models with images, processes images and generates pixel_values.
        """
        # Extract sequences
        input_ids_list = [x["input_ids"] for x in batch]
        labels_list = [x["labels"] for x in batch]
        
        # Check if this is packed data (has sequence_lengths field)
        has_packing = "sequence_lengths" in batch[0]
        sequence_lengths_list = [x["sequence_lengths"] for x in batch] if has_packing else None
        
        # Check if this is VL data (image bytes stored externally via archives)
        # 使用 get() is not None 而非 in 检查，因为 concatenate_datasets() 合并不同 schema 的数据集时
        # 会自动为缺失字段填充 None，导致字段存在但值为 None 的情况
        has_single_image_ref_field = batch[0].get("image_ref") is not None
        has_packed_image_ref_field = batch[0].get("image_refs") is not None
        has_images = self.is_vl_model and (
            has_single_image_ref_field
            or has_packed_image_ref_field
        )
        
        # Apply dynamic truncation if max_length is specified
        if self.max_length is not None:
            input_ids_list = [ids[:self.max_length] for ids in input_ids_list]
            labels_list = [lbls[:self.max_length] for lbls in labels_list]
        
        # Determine padding length
        if self.pad_to_max_length and self.max_length is not None:
            # Pad to max_length for testing memory peak
            max_len = self.max_length
        else:
            # Dynamic padding: pad to longest sequence in batch (more efficient)
            max_len = max(len(ids) for ids in input_ids_list)
        
        # Pad sequences to max length
        batch_size = len(batch)
        pad_token_id = self.tokenizer.pad_token_id
        
        # Initialize tensors
        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 is ignored in loss
        
        # Fill in the actual values
        for i, (ids, lbls) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(lbls, dtype=torch.long)
        
        # Process images for VL models FIRST to determine if we have actual images
        # This is needed because position_ids handling differs based on whether images are present
        actual_has_images = False
        pixel_values = None
        image_grid_thw = None
        
        if has_images and self.processor is not None:
            # Check if this is packed data with images (packed VL data uses archived refs)
            is_packed_vl = has_packing and has_packed_image_ref_field
            
            if is_packed_vl:
                # Packed VL mode: each sample may have multiple images from multiple sub-sequences
                all_images = []
                image_counts = []  # Number of images per packed sample
                
                for x in batch:
                    sample_refs = x.get("image_refs", [])
                    sample_bases = x.get("image_bases", [])
                    valid_imgs = []
                    for ref, base_key in zip(sample_refs, sample_bases):
                        img = self._load_image_from_archive(ref, base_key)
                        if img is not None:
                            valid_imgs.append(img)
                    all_images.extend(valid_imgs)
                    image_counts.append(len(valid_imgs))
                
                if all_images:
                    actual_has_images = True
                    try:
                        image_inputs = self.processor.image_processor(
                            images=all_images,
                            return_tensors="pt"
                        )
                        pixel_values = image_inputs.get("pixel_values")
                        if "image_grid_thw" in image_inputs:
                            image_grid_thw = image_inputs["image_grid_thw"]
                    except Exception as e:
                        logger.warning(f"Failed to process packed images: {e}")
                        actual_has_images = False
                else:
                    # 直接报错：数据集声称有图像但全部加载失败
                    total_refs = sum(len(x.get("image_refs", [])) for x in batch)
                    if total_refs > 0:
                        raise RuntimeError(
                            f"图像加载失败：本 batch 包含 {total_refs} 个图像引用，但全部加载失败！\n"
                            f"请检查 image_archives_info.json 是否存在于数据集目录中。\n"
                            f"可使用 scripts/fix_multiview_archives_info.py 脚本修复缺失的 image_archives_info.json 文件。"
                        )
            else:
                # Non-packed VL mode: standard image processing
                images = []
                has_image_flags = []
                valid_image_indices = []  # Track which samples have valid images
                
                for idx, x in enumerate(batch):
                    img = None
                    image_ref = x.get("image_ref")
                    base_key = x.get("image_base")
                    has_img = x.get("has_image", bool(image_ref))
                    if image_ref and base_key:
                        img = self._load_image_from_archive(image_ref, base_key)
                    has_image_flags.append(has_img)
                    if img is not None:
                        images.append(img)
                        valid_image_indices.append(idx)
                
                # Process images with the VL processor
                valid_images = images
                if valid_images:
                    actual_has_images = True
                    try:
                        image_inputs = self.processor.image_processor(
                            images=valid_images,
                            return_tensors="pt"
                        )
                        pixel_values = image_inputs.get("pixel_values")
                        if "image_grid_thw" in image_inputs:
                            image_grid_thw = image_inputs["image_grid_thw"]
                    except Exception as e:
                        logger.warning(f"Failed to process images: {e}")
                        actual_has_images = False
        
        # Generate position_ids for attention isolation in packed sequences
        # 
        # 🔑 关键理解 (Qwen2-VL/Qwen3-VL):
        # 1. Tokenize 时，图像会被预先放置 <|image_pad|> token（数量根据 image_grid_thw 确定）
        # 2. Forward 时，pixel_values 转换为 visual embeddings 替换 <|image_pad|>
        # 3. 因此序列长度在 tokenize 后就确定了，可以提前计算 position_ids
        # 4. 通过手动计算正确的 3D position_ids，可以实现 VL 模型的 packing 支持
        #
        if has_packing:
            base_position_ids = build_varlen_position_ids(
                sequence_lengths_list,
                max_len,
                input_ids.device,
            )

            if self.is_vl_model:
                if actual_has_images and image_grid_thw is not None:
                    # ✅ VL 模型 + 有图像 + packing：计算正确的 3D position_ids
                    # 使用 compute_vl_position_ids 函数来生成包含图像位置信息的 position_ids
                    # 参考 Qwen3VLModel.get_rope_index() 的实现逻辑
                    # 从 model.config 获取特殊 token IDs 和 spatial_merge_size
                    # 参考: modeling_qwen3_vl.py 第 931-933 行
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    model_config = unwrapped_model.config

                    image_token_id = model_config.image_token_id
                    vision_start_token_id = model_config.vision_start_token_id
                    spatial_merge_size = model_config.vision_config.spatial_merge_size

                    vl_position_ids = compute_vl_position_ids(
                        input_ids=input_ids,
                        image_grid_thw=image_grid_thw,
                        spatial_merge_size=spatial_merge_size,
                        image_token_id=image_token_id,
                        vision_start_token_id=vision_start_token_id,
                        sequence_lengths=sequence_lengths_list,
                    )
                else:
                    # VL 模型 + 纯文本 packing：构造简单的 3D position_ids
                    vl_position_ids = torch.zeros((3, batch_size, max_len), dtype=torch.long, device=input_ids.device)
                    for i, seq_lengths in enumerate(sequence_lengths_list):
                        offset = 0
                        for length in seq_lengths:
                            span = min(length, max_len - offset)
                            if span <= 0:
                                break
                            positions = torch.arange(span, device=input_ids.device)
                            vl_position_ids[0, i, offset:offset + span] = positions
                            vl_position_ids[1, i, offset:offset + span] = positions
                            vl_position_ids[2, i, offset:offset + span] = positions
                            offset += span
                        if offset < max_len:
                            pad_span = max_len - offset
                            pad_positions = torch.arange(pad_span, device=input_ids.device)
                            vl_position_ids[0, i, offset:] = pad_positions
                            vl_position_ids[1, i, offset:] = pad_positions
                            vl_position_ids[2, i, offset:] = pad_positions

                position_ids = torch.cat(
                    [base_position_ids.unsqueeze(0), vl_position_ids],
                    dim=0,
                )
                logger.debug(
                    f"VL packing enabled: computed stacked position_ids with shape {position_ids.shape}"
                )
            else:
                # 普通 LLM 使用 1D position_ids
                position_ids = base_position_ids

            # 🔑 对于 packing 模式，不传 attention_mask
            result = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids
            }
        else:
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        # Add image data to result if present
        if actual_has_images:
            if pixel_values is not None:
                result["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                result["image_grid_thw"] = image_grid_thw
            
            # Add additional tracking info for packed VL mode
            if has_packing and has_packed_image_ref_field:
                result["image_counts"] = torch.tensor(image_counts, dtype=torch.long)
            elif not has_packing:
                result["image_sample_indices"] = torch.tensor(valid_image_indices, dtype=torch.long)
                result["has_image"] = torch.tensor(has_image_flags, dtype=torch.bool)
        return result
    
    def _create_optimizer(self):
        """Create optimizer."""
        optimizer_type = self.config.get("optimizer", "adamw")
        learning_rate = self.config.get("learning_rate", 2e-5)
        weight_decay = self.config.get("weight_decay", 0.01)
        
        # Get parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=self.config.get("adam_betas", (0.9, 0.999)),
                eps=self.config.get("adam_epsilon", 1e-8)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return optimizer
    
    def _apply_dummy_gradient(self, loss: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        [Qwen3-VL 修复补丁]
        解决纯文本 Batch 导致 Visual Tower 无梯度从而导致死锁的问题。
        
        当 VL 模型处理纯文本 batch（无图像输入）时，Visual Tower 不参与前向计算，
        因此在 backward 时没有梯度。这会导致 FSDP 在同步梯度时发生死锁。
        
        解决方案：对于纯文本 batch，构造一个 dummy 输入通过 Visual Tower，
        并将其输出乘以 0.0 加到 loss 上，从而保持计算图连接但不影响 loss 数值。
        
        Args:
            loss: 当前 batch 的 loss
            batch: 当前 batch 的输入数据
            
        Returns:
            修正后的 loss（对于纯文本 batch，包含 dummy 梯度连接）
        """
        if not self.is_vl_model:
            return loss
        
        # 检查是否有视觉输入
        has_visual_input = (
            ("pixel_values" in batch and batch["pixel_values"] is not None) or 
            ("pixel_values_videos" in batch and batch["pixel_values_videos"] is not None)
        )
        
        if has_visual_input:
            return loss
        
        # --- A. 获取 Visual Module (不解包，保留分布式环境) ---
        if hasattr(self.model, "visual"):
            visual_module = self.model.visual
        elif hasattr(self.model, "module") and hasattr(self.model.module, "visual"):
            visual_module = self.model.module.visual
        else:
            raise AttributeError("Could not find 'visual' module in model or model.module")
        
        # --- B. 准备 Dummy 数据 (必须与 loss 同设备、同精度) ---
        # Qwen2/3-VL 的 patch_size 通常是 14 或 16
        patch_size = 16
        if hasattr(visual_module, "config"):
            patch_size = getattr(visual_module.config, "patch_size", 16)
        
        # 构造最小 Batch: in_channels * temporal * h * w
        dummy_dim = 3 * 2 * patch_size * patch_size
        
        # dtype 和 device 必须跟随 loss
        dummy_pixel_values = torch.zeros(
            (1, dummy_dim), 
            dtype=loss.dtype, 
            device=loss.device
        ).requires_grad_(True)
        
        dummy_grid_thw = torch.tensor(
            [[1, patch_size, patch_size]], 
            dtype=torch.long, 
            device=loss.device
        )
        
        # --- C. 强制 Forward ---
        dummy_out, dummy_out2_lists = visual_module(
            hidden_states=dummy_pixel_values,
            grid_thw=dummy_grid_thw
        )
        
        # --- D. 注入梯度 ---
        # 乘以 0.0 避免影响 Loss 数值，但保留计算图连接
        loss = loss + dummy_out.sum() * 0.0
        loss = loss + sum([out.sum() * 0.0 for out in dummy_out2_lists])
        
        return loss
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_epochs = self.config.get("num_epochs", 3)
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / gradient_accumulation_steps
        )
        max_train_steps = num_epochs * num_update_steps_per_epoch
        
        warmup_steps = self.config.get("warmup_steps", 0)
        if warmup_steps == 0:
            warmup_ratio = self.config.get("warmup_ratio", 0.1)
            warmup_steps = int(max_train_steps * warmup_ratio)
        
        scheduler_type = self.config.get("lr_scheduler_type", "cosine")
        
        lr_scheduler = get_scheduler(
            name=scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps
        )
        
        return lr_scheduler
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config.get("num_epochs", 3)
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        logging_steps = self.config.get("logging_steps", 10)
        save_steps = self.config.get("save_steps", 500)
        eval_steps = self.config.get("eval_steps", 500)
        
        if self.accelerator.is_main_process:
            logger.info("Starting training")
            logger.info(f"Num epochs: {num_epochs}")
            logger.info(f"Total batches per epoch: {len(self.train_dataloader)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.model.train()
            
            total_loss = 0
            num_loss_steps = 0  # Track actual number of batches for accurate averaging
            progress_bar = tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch + 1}/{num_epochs}"
            )
            
            for step, batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    # 应用 VL 模型 FSDP 修复补丁（处理纯文本 batch 时 Visual Tower 无梯度的问题）
                    loss = self._apply_dummy_gradient(loss, batch)
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # 只在真正的 optimizer step 时更新 learning rate scheduler
                # 避免在 gradient accumulation 期间多次调用导致 warmup 过早结束
                if self.accelerator.sync_gradients:
                    self.lr_scheduler.step()
                
                total_loss += loss.detach().float()
                num_loss_steps += 1  # Count every batch
                
                if self.accelerator.sync_gradients:
                    self.accelerator.wait_for_everyone()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % logging_steps == 0:
                        avg_loss = total_loss / num_loss_steps
                        
                        # Gather loss from all GPUs and compute mean
                        avg_loss = self.accelerator.gather(avg_loss).mean()
                        
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{current_lr:.2e}"
                        })
                        
                        if self.accelerator.is_main_process:
                            logger.info(
                                f"Step {self.global_step}: "
                                f"loss={avg_loss:.4f}, "
                                f"lr={current_lr:.2e}"
                            )
                            
                            # Log to TensorBoard/WandB
                            self.accelerator.log({
                                "train/loss": avg_loss.item() if torch.is_tensor(avg_loss) else float(avg_loss),
                                "train/learning_rate": current_lr,
                                "train/epoch": epoch,
                                "train/global_step": self.global_step
                            }, step=self.global_step)
                        
                        total_loss = 0
                        num_loss_steps = 0
                    
                    # Evaluation
                    if eval_steps > 0 and self.global_step % eval_steps == 0:
                        if self.eval_dataloader is not None:
                            self.evaluate()
                            self.model.train()
                    
                    # Saving
                    if save_steps > 0 and self.global_step % save_steps == 0:
                        self.save_checkpoint()
                    
                    # 检查是否收到中断信号（同步到所有节点）
                    # 使用分布式通信确保所有节点同时进入保存流程
                    if torch.distributed.is_initialized():
                        interrupted_tensor = torch.tensor([1 if self._interrupted else 0], device=self.accelerator.device)
                        torch.distributed.all_reduce(interrupted_tensor, op=torch.distributed.ReduceOp.MAX)
                        should_interrupt = interrupted_tensor.item() > 0
                    else:
                        should_interrupt = self._interrupted
                    
                    if should_interrupt:
                        if self.accelerator.is_main_process:
                            logger.info(f"在 step {self.global_step} 处中断训练，正在保存模型...")
                        self.save_checkpoint(interrupted=True)
                        if self.accelerator.is_main_process:
                            logger.info("模型已保存，训练中断退出")
                        # 恢复原始信号处理器
                        signal.signal(signal.SIGINT, self._original_sigint_handler)
                        self.accelerator.end_training()
                        return
        
        # Final save
        if self.accelerator.is_main_process:
            logger.info("Training completed")
        
        self.save_checkpoint(final=True)
        
        # 恢复原始信号处理器
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        
        # End tracking
        self.accelerator.end_training()
    
    def evaluate(self):
        """Evaluation loop."""
        if self.eval_dataloader is None:
            return
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process
            ):
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                num_batches += 1
        
        # Gather losses from all GPUs
        total_loss = self.accelerator.gather(total_loss).sum()
        num_batches = self.accelerator.gather(torch.tensor(num_batches, device=self.accelerator.device)).sum()
        
        # Calculate average loss across all GPUs and all batches
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(avg_loss)
        
        if self.accelerator.is_main_process:
            logger.info(
                f"Eval at step {self.global_step}: "
                f"loss={avg_loss:.4f}, perplexity={perplexity:.2f}"
            )
            
            # Log to TensorBoard/WandB
            self.accelerator.log({
                "eval/loss": avg_loss.item() if torch.is_tensor(avg_loss) else float(avg_loss),
                "eval/perplexity": perplexity.item() if torch.is_tensor(perplexity) else float(perplexity),
            }, step=self.global_step)
        
        self.accelerator.wait_for_everyone()
    
    def save_checkpoint(self, final: bool = False, interrupted: bool = False):
        """Save model checkpoint with full training state.
        
        Args:
            final: Whether this is the final checkpoint after training completes
            interrupted: Whether this is a checkpoint saved due to Ctrl+C interruption
        """
        # 标记正在保存，允许第二次 Ctrl+C 跳过保存
        self._saving = True
        
        output_dir = Path(self.config["output_dir"])
        
        if final:
            save_dir = output_dir / "final"
        elif interrupted:
            save_dir = output_dir / f"checkpoint-{self.global_step}-interrupted"
        else:
            save_dir = output_dir / f"checkpoint-{self.global_step}"
        
        if self.accelerator.is_main_process:
            logger.info(f"Saving checkpoint to {save_dir}")
        
        self.accelerator.wait_for_everyone()
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Get state dict (collective operation, all processes must participate)
        state_dict = self.accelerator.get_state_dict(self.model)
        
        # Save model
        if self.accelerator.is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_model.save_pretrained(
                save_dir,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=state_dict
            )
            
            # Save tokenizer or processor
            if self.processor is not None:
                self.processor.save_pretrained(save_dir)
            else:
                self.tokenizer.save_pretrained(save_dir)
        
        # Get optimizer and scheduler state (may need all processes in FSDP)
        optimizer_state = self.optimizer.state_dict()
        lr_scheduler_state = self.lr_scheduler.state_dict()
        
        if self.accelerator.is_main_process:
            # Save training state (optimizer, scheduler, RNG states)
            training_state = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "config": self.config,
                "optimizer_state": optimizer_state,
                "lr_scheduler_state": lr_scheduler_state,
                "rng_state": torch.get_rng_state(),
            }
            
            # Save CUDA RNG state if available
            if torch.cuda.is_available():
                training_state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
            
            torch.save(
                training_state,
                save_dir / "training_state.pt"
            )
            
            logger.info(f"Checkpoint saved successfully to {save_dir}")
        
        # Clean up old checkpoints if save_total_limit is set
        if not final:
            self._cleanup_checkpoints()
        
        self.accelerator.wait_for_everyone()
        
        # 标记保存完成
        self._saving = False
    
    def load_weights_only(self, checkpoint_path: str):
        """
        仅加载模型权重，不恢复训练状态（optimizer、lr_scheduler、epoch等）。
        适用于将已训练的 checkpoint 作为新基座进行微调。
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        if self.accelerator.is_main_process:
            logger.info(f"Initializing model weights from {checkpoint_path} (training state NOT restored)")
        
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        adapter_path = checkpoint_path / "adapter_model.safetensors"
        adapter_weights = load_file(adapter_path)
        set_peft_model_state_dict(unwrapped_model, adapter_weights)
        
        if self.accelerator.is_main_process:
            logger.info(f"Model weights loaded. Optimizer and LR scheduler will start fresh.")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        if self.accelerator.is_main_process:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # 加载模型权重（使用 set_peft_model_state_dict 直接设置权重，保持参数ID不变）
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        adapter_path = checkpoint_path / "adapter_model.safetensors"
        adapter_weights = load_file(adapter_path)
        set_peft_model_state_dict(unwrapped_model, adapter_weights)
        if self.accelerator.is_main_process:
            logger.info(f"Model weights restored from {adapter_path}")
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location="cpu")
            
            # Restore training progress
            self.epoch = training_state.get("epoch", 0)
            self.global_step = training_state.get("global_step", 0)
            
            # Restore optimizer state
            if "optimizer_state" in training_state:
                self.optimizer.load_state_dict(training_state["optimizer_state"])
                if self.accelerator.is_main_process:
                    logger.info("Optimizer state restored")
            
            # Restore learning rate scheduler state
            if "lr_scheduler_state" in training_state:
                self.lr_scheduler.load_state_dict(training_state["lr_scheduler_state"])
                if self.accelerator.is_main_process:
                    logger.info("Learning rate scheduler state restored")
            
            # Restore RNG states for reproducibility
            if "rng_state" in training_state:
                torch.set_rng_state(training_state["rng_state"])
            
            if "cuda_rng_state" in training_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(training_state["cuda_rng_state"])
            
            if self.accelerator.is_main_process:
                logger.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")
        else:
            logger.warning(f"Training state file not found: {training_state_path}")
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints to keep only the most recent ones."""
        save_total_limit = self.config.get("save_total_limit", None)
        
        if save_total_limit is None or save_total_limit <= 0:
            return
        
        if not self.accelerator.is_main_process:
            return
        
        output_dir = Path(self.config["output_dir"])
        
        # Find all checkpoint directories
        checkpoints = []
        for path in output_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-"):
                try:
                    step = int(path.name.split("-")[1])
                    checkpoints.append((step, path))
                except (IndexError, ValueError):
                    continue
        
        # Sort by step number (newest first)
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints beyond the limit
        for _, checkpoint_path in checkpoints[save_total_limit:]:
            logger.info(f"Removing old checkpoint: {checkpoint_path}")
            import shutil
            shutil.rmtree(checkpoint_path)
    
    def _save_config(self):
        """Save training configuration to output directory for record keeping."""
        import json
        import yaml
        from datetime import datetime
        
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full config as YAML
        config_save_path = output_dir / "training_config.yaml"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Training config saved to {config_save_path}")
        
        # Save a JSON summary with timestamp for easy parsing
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config_file": self.config.get("_config_file_path", "unknown"),
            "model_name": self.config.get("model_name"),
            "training_mode": self.config.get("training_mode"),
            "output_dir": self.config.get("output_dir"),
            "num_epochs": self.config.get("num_epochs"),
            "learning_rate": self.config.get("learning_rate"),
            "per_device_train_batch_size": self.config.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": self.config.get("gradient_accumulation_steps"),
            "max_length": self.config.get("max_length"),
            "seed": self.config.get("seed"),
        }
        
        # Add dataset info
        if "batch_dirs" in self.config:
            batch_config = self.config["batch_dirs"]
            summary["dataset_type"] = "batch_dirs"
            # Handle both single dict and list of dicts format
            if isinstance(batch_config, dict):
                summary["batch_dirs"] = [batch_config]
            else:
                summary["batch_dirs"] = batch_config
        elif "dataset_path" in self.config:
            summary["dataset_type"] = "single_file"
            summary["dataset_path"] = self.config.get("dataset_path")
        
        # Add LoRA specific info if applicable
        if self.config.get("training_mode") == "lora":
            lora_config = self.config.get("lora_config", {})
            summary["lora_r"] = lora_config.get("r")
            summary["lora_alpha"] = lora_config.get("lora_alpha")
            summary["lora_dropout"] = lora_config.get("lora_dropout")
        
        # Add VL model info
        summary["is_vl_model"] = self.is_vl_model
        
        summary_path = output_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Training summary saved to {summary_path}")

    def _log_lora_target_modules(self) -> None:
        """Log and persist the list of modules that received LoRA adapters."""
        if self.config.get("training_mode") != "lora":
            return
        if not self.accelerator.is_main_process:
            return
        output_dir = Path(self.config.get("output_dir", "./outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "lora_target_modules.txt"
        try:
            from peft.tuners.lora import LoraLayer
        except ImportError as exc:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"Unable to inspect LoRA layers (peft import failed): {exc}\n")
            return

        lora_module_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                lora_module_names.append(name)

        lora_module_names = sorted(set(lora_module_names))
        if not lora_module_names:
            content = "LoRA training requested but no LoRA layers were detected.\n"
        else:
            content = "\n".join(lora_module_names) + "\n"

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:
            # 即便写入失败也不要打印到命令行，因此静默失败
            pass

    def _log_all_model_layers(self) -> None:
        """Log and persist the list of all model layers/modules."""
        if not self.accelerator.is_main_process:
            return
        output_dir = Path(self.config.get("output_dir", "./outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "all_model_layers.txt"

        all_module_info = []
        for name, module in self.model.named_modules():
            # 获取模块类型名称
            module_type = type(module).__name__
            # 获取模块参数数量
            num_params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            
            if name == "":
                name = "(root)"
            
            info_line = f"{name} | {module_type} | params: {num_params:,} | trainable: {trainable_params:,}"
            all_module_info.append(info_line)

        # 统计总参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        header = [
            "=" * 80,
            "Model Architecture - All Layers",
            "=" * 80,
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {total_trainable:,}",
            f"Trainable Ratio: {total_trainable / total_params * 100:.2f}%" if total_params > 0 else "N/A",
            "=" * 80,
            "Format: module_name | module_type | params (direct) | trainable (direct)",
            "=" * 80,
            ""
        ]
        
        content = "\n".join(header + all_module_info) + "\n"

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"All model layers saved to {save_path}")
        except Exception as exc:
            # 即便写入失败也不要打印到命令行，因此静默失败
            pass

