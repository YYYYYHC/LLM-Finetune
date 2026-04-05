"""
vLLM-based inference pipeline for generating JSON scenes.

This module loads either full fine-tuned checkpoints or LoRA adapters and
runs high-throughput inference with automatic JSON extraction.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
import hashlib
import itertools
import json
import pickle
import subprocess
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.utils.logger import setup_logger


@dataclass
class SampleInput:
    """Container for a single inference sample."""

    uid: str
    messages: List[Dict[str, Any]]
    image_path: Optional[Path]
    prompt_preview: str
    image_data: Optional[Image.Image] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    condition_type: str = "text"
    condition_value: Optional[Any] = None
    # multi_view 模式：多张图片
    image_paths: Optional[List[Path]] = None
    image_data_list: Optional[List[Image.Image]] = None
    # 从 arrow 加载时的 ground truth（仅 panorama/multi_view 模式）
    ground_truth: Optional[str] = None


class MessageBuilder:
    """Create chat messages aligned with training templates."""

    def __init__(self, mode: str, image_root: Optional[str] = None):
        self.mode = mode
        self.image_root = Path(image_root).resolve() if image_root else None

    def build(self, record: Dict[str, Any]) -> SampleInput:
        messages = record.get("messages")
        image_path = record.get("image_path")
        image_data = record.get("image_data") or record.get("image_obj")
        # multi_view 模式支持
        image_paths = record.get("image_paths")  # List[str]
        image_data_list = record.get("image_data_list")  # List[Image]
        condition_payload: Optional[Any] = None

        if messages is None:
            messages, image_path, condition_payload = self._from_mode(record)
        else:
            image_path = image_path or record.get("image") or record.get("image_file")
            condition_payload = self._infer_condition_payload(record)
            # 对于从 arrow 加载的 blueprint 数据，直接使用 record 中的 blueprint
            if self.mode == "blueprint" and condition_payload is None:
                condition_payload = record.get("blueprint")

        resolved_image = self._resolve_image_path(image_path, record)
        # 解析多图片路径（multi_view 模式）
        resolved_images: Optional[List[Path]] = None
        if image_paths:
            resolved_images = [self._resolve_image_path(p, record) for p in image_paths]
            resolved_images = [p for p in resolved_images if p is not None]
        
        # 判断是否有图片
        num_images = 0
        if resolved_images:
            num_images = len(resolved_images)
        elif image_data_list:
            num_images = len(image_data_list)
        elif resolved_image is not None or image_data is not None:
            num_images = 1
        
        has_image = num_images > 0
        prepared_messages = self._prepare_messages(messages, has_image, num_images)
        prompt_preview = self._extract_prompt_preview(prepared_messages)
        uid = self._resolve_uid(record)
        metadata = {
            "source_path": record.get("__source_path"),
            "extra_tags": record.get("tags")
        }

        condition_type = self._condition_type()
        condition_value = self._finalize_condition_value(
            condition_type,
            condition_payload,
            prompt_preview,
            record,
            resolved_image,
            resolved_images
        )

        return SampleInput(
            uid=uid,
            messages=prepared_messages,
            image_path=resolved_image,
            prompt_preview=prompt_preview,
            image_data=image_data,
            metadata={k: v for k, v in metadata.items() if v is not None},
            condition_type=condition_type,
            condition_value=condition_value,
            image_paths=resolved_images,
            image_data_list=image_data_list,
            ground_truth=record.get("ground_truth")
        )

    def _from_mode(self, record: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Optional[str], Optional[Any]]:
        if self.mode == "unconditional":
            prompt = record.get("prompt") or record.get("instruction")
            if not prompt:
                prompt = "Generate a 3D scene in JSON format:"
            return ([{"role": "user", "content": prompt}]), None, prompt

        if self.mode == "caption":
            caption = record.get("caption") or record.get("description") or record.get("prompt")
            if not caption:
                raise ValueError("caption mode requires 'caption' or 'description'")
            content = f"Generate a 3D scene in JSON format based on the following description:\n{caption}"
            return ([{"role": "user", "content": content}]), None, content

        if self.mode == "blueprint":
            blueprint = record.get("blueprint") or record.get("plan")
            if blueprint is None:
                raise ValueError("blueprint mode requires 'blueprint' or 'plan'")
            if not isinstance(blueprint, dict):
                raise ValueError("blueprint data must be a dictionary")
            blueprint_str = json.dumps(blueprint, ensure_ascii=False, separators=(",", ":"))
            content = f"Generate 3D objects for the following room blueprint:\n{blueprint_str}"
            return ([{"role": "user", "content": content}]), None, blueprint

        if self.mode == "panorama":
            user_prompt = record.get("prompt")
            if not user_prompt:
                user_prompt = "Based on this panorama image, generate a 3D scene in JSON format:"
            print(f"panorama mode, user_prompt: {user_prompt}")
            image_path = record.get("image_path") or record.get("image") or record.get("image_file")
            if not image_path:
                raise ValueError("panorama mode requires 'image_path'")
            return ([{"role": "user", "content": user_prompt}]), image_path, image_path

        if self.mode == "multi_view":
            user_prompt = record.get("prompt")
            if not user_prompt:
                user_prompt = "Based on these multi-view images, generate a 3D scene in JSON format:"
            print(f"multi_view mode, user_prompt: {user_prompt}")
            # multi_view 模式下 image_path 返回 None，使用 image_paths
            return ([{"role": "user", "content": user_prompt}]), None, None

        raise ValueError(f"Unsupported mode: {self.mode}")

    def _prepare_messages(self, messages: List[Dict[str, Any]], has_image: bool, num_images: int = 1) -> List[Dict[str, Any]]:
        prepared = []
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")

            if has_image:
                if idx == 0 and role == "user":
                    if isinstance(content, list):
                        prepared.append(msg)
                        continue
                    # 构建多图片的 content（每张图片一个 image 类型）
                    image_items = [{"type": "image"} for _ in range(num_images)]
                    prepared.append({
                        "role": "user",
                        "content": image_items + [{"type": "text", "text": content}]
                    })
                    continue

                if isinstance(content, list):
                    prepared.append(msg)
                else:
                    prepared.append({"role": role, "content": [{"type": "text", "text": content}]})
            else:
                if isinstance(content, list):
                    prepared.append(msg)
                else:
                    prepared.append({"role": role, "content": content})

        return prepared

    def _extract_prompt_preview(self, messages: Sequence[Dict[str, Any]]) -> str:
        if not messages:
            return ""
        first = messages[0]
        content = first.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [chunk.get("text", "") for chunk in content if isinstance(chunk, dict)]
            return " ".join(filter(None, texts))
        return ""

    def _resolve_uid(self, record: Dict[str, Any]) -> str:
        if "id" in record and record["id"]:
            return str(record["id"])
        if "filename" in record and record["filename"]:
            return Path(record["filename"]).stem
        return f"sample_{record.get('__row_id', 0):06d}"

    def _resolve_image_path(self, image_path: Optional[str], record: Dict[str, Any]) -> Optional[Path]:
        if not image_path:
            return None
        candidate = Path(image_path)
        if not candidate.is_absolute():
            base_dir = record.get("__source_dir")
            if base_dir:
                candidate = Path(base_dir) / candidate
            elif self.image_root:
                candidate = self.image_root / candidate
        resolved = candidate.expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"Image file not found: {resolved}")
        return resolved

    def _condition_type(self) -> str:
        if self.mode == "blueprint":
            return "blueprint"
        if self.mode == "panorama":
            return "panorama"
        if self.mode == "multi_view":
            return "multi_view"
        return "text"

    def _infer_condition_payload(self, record: Dict[str, Any]) -> Optional[Any]:
        if self.mode == "blueprint":
            return record.get("blueprint") or record.get("plan")
        if self.mode == "panorama":
            return record.get("image_path") or record.get("image") or record.get("image_file")
        if self.mode == "multi_view":
            return record.get("image_paths")
        if self.mode == "caption":
            return (
                record.get("caption")
                or record.get("description")
                or record.get("prompt")
            )
        if self.mode == "unconditional":
            return record.get("prompt") or record.get("instruction")
        return None

    def _finalize_condition_value(
        self,
        condition_type: str,
        raw_condition: Optional[Any],
        prompt_preview: str,
        record: Dict[str, Any],
        resolved_image: Optional[Path],
        resolved_images: Optional[List[Path]] = None
    ) -> Optional[Any]:
        if condition_type == "text":
            candidate = raw_condition or prompt_preview
            if candidate:
                return str(candidate)
            raise ValueError("Text condition is missing for the provided record")
        if condition_type == "blueprint":
            blueprint = raw_condition or record.get("blueprint") or record.get("plan")
            if not isinstance(blueprint, dict):
                raise ValueError("Blueprint condition must be provided as a dictionary")
            return blueprint
        if condition_type == "panorama":
            return resolved_image
        if condition_type == "multi_view":
            return resolved_images
        return None


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for idx in range(0, len(seq), size):
        yield seq[idx: idx + size]


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_json_from_text(text: str) -> tuple[Optional[Any], Optional[str], Optional[str]]:
    """直接解析 JSON 文本，假设模型输出为纯 JSON 格式。"""
    cleaned = text.strip()
    # 移除 <think>...</think> 标签
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1].strip()
    if not cleaned:
        return None, None, "empty_response"
    try:
        return json.loads(cleaned), cleaned, None
    except json.JSONDecodeError as exc:
        return None, cleaned, f"json_error: {exc}"


def _strip_chat_prompt_markers(text: str, tokenizer: AutoTokenizer) -> str:
    """Remove chat-template markers to recover the raw user prompt."""
    cleaned = text.strip()
    if "<|im_start|>user" in cleaned:
        cleaned = cleaned.split("<|im_start|>user", 1)[-1]
    if "<|im_start|>assistant" in cleaned:
        cleaned = cleaned.split("<|im_start|>assistant", 1)[0]
    if "<|im_end|>" in cleaned:
        cleaned = cleaned.split("<|im_end|>", 1)[0]

    # Remove VL placeholders and residual special tokens
    for marker in ("<|vision_start|>", "<|vision_end|>"):
        cleaned = cleaned.replace(marker, "")
    cleaned = cleaned.replace("<|image_pad|>", "")
    cleaned = cleaned.replace("</s>", "")
    for special in {tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token, "<|endoftext|>"}:
        if special:
            cleaned = cleaned.replace(special, "")
    return cleaned.strip()


def _decode_user_prompt(tokenizer: AutoTokenizer, input_ids: List[int], labels: List[int]) -> Optional[str]:
    """Extract and sanitize the user prompt from packed token/label arrays."""
    if not input_ids:
        return None
    first_assistant = next((idx for idx, label in enumerate(labels) if label != -100), len(input_ids))
    user_ids = input_ids[:first_assistant] or input_ids
    raw = tokenizer.decode(user_ids, skip_special_tokens=False)
    prompt = _strip_chat_prompt_markers(raw, tokenizer)
    return prompt or raw.strip()


def _decode_assistant_response(tokenizer: AutoTokenizer, input_ids: List[int], labels: List[int]) -> Optional[str]:
    """Extract assistant response (ground truth) from packed token/label arrays."""
    if not input_ids:
        return None
    first_assistant = next((idx for idx, label in enumerate(labels) if label != -100), len(input_ids))
    if first_assistant >= len(input_ids):
        return None
    assistant_ids = input_ids[first_assistant:]
    raw = tokenizer.decode(assistant_ids, skip_special_tokens=True)
    return raw.strip() if raw else None


def _visualize_blueprint(data: Union[str, dict], output_path: Path) -> None:
    """Visualize blueprint rooms as polygons.
    
    Args:
        data: JSON string or dict containing blueprint info
        output_path: Path to save the visualization image
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        blueprint = data.get("blueprint", {})
        rooms = blueprint.get("rooms", {})
        if not rooms:
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.tab10.colors
        
        for idx, (name, room) in enumerate(rooms.items()):
            coords = room.get("shape", {}).get("coordinates", [[]])[0]
            if len(coords) < 3:
                continue
            poly = MplPolygon(coords, closed=True, facecolor=colors[idx % 10], 
                            edgecolor='black', alpha=0.5, linewidth=2)
            ax.add_patch(poly)
            # 标注房间名称（取 semantics 或 name）
            xs, ys = zip(*coords)
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            label = room.get("semantics", name.split("/")[0])
            ax.text(cx, cy, label, ha='center', va='center', fontsize=9)
        
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Blueprint Rooms')
        ax.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        pass  # 静默忽略可视化错误


class ArrowImageExtractor:
    """Materialize panorama images stored in tar archives for inference."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path).resolve()
        self.info_path = self._locate_info_file(self.dataset_path)
        if self.info_path is None:
            raise FileNotFoundError(
                f"image_archives_info.json not found near {self.dataset_path}; "
                "please point --arrow_dir to a batch directory created by prepare_data.sh"
            )
        with open(self.info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        relpath = info.get("archives_relpath", "")
        if relpath:
            self.archives_root = (self.info_path.parent / relpath).resolve()
        else:
            self.archives_root = self.info_path.parent
        self._cache: Dict[str, Path] = {}
        self._temp_dir = tempfile.TemporaryDirectory(prefix="arrow_images_")
        self._temp_path = Path(self._temp_dir.name)
        # 索引缓存：archive_path -> {member_name: (offset, size)}
        self._tar_index_cache: Dict[str, Dict[str, tuple]] = {}
        # 文件句柄缓存：archive_path -> file handle (用于索引访问时的 seek/read)
        self._tar_file_cache: Dict[str, Any] = {}
        atexit.register(self._cleanup)

    @staticmethod
    def _locate_info_file(start: Path) -> Optional[Path]:
        current = start
        for _ in range(3):
            candidate = current / "image_archives_info.json"
            if candidate.exists():
                return candidate
            current = current.parent
        return None

    def _cleanup(self) -> None:  # pragma: no cover - best effort cleanup
        # 关闭所有打开的文件句柄
        for fp in self._tar_file_cache.values():
            try:
                fp.close()
            except Exception:
                pass
        self._tar_file_cache.clear()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def materialize(self, image_ref: str) -> Path:
        if not image_ref:
            raise ValueError("image_ref is required for panorama samples")
        if image_ref in self._cache:
            return self._cache[image_ref]
        if "::" not in image_ref:
            raise ValueError(f"Invalid image_ref format: {image_ref}")
        archive_rel, member_name = image_ref.split("::", 1)
        archive_path = (self.archives_root / archive_rel).resolve()
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        cache_key = str(archive_path)
        
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
        
        # 使用索引进行 O(1) 访问
        if index is not None and member_name in index:
            offset, size = index[member_name]
            fp = self._tar_file_cache.get(cache_key)
            if fp is None:
                fp = open(archive_path, "rb")
                self._tar_file_cache[cache_key] = fp
            fp.seek(offset)
            data = fp.read(size)
        else:
            # 回退：无索引时使用 tarfile（兼容旧数据）
            with tarfile.open(archive_path, "r") as tar:
                member = tar.extractfile(member_name)
                if member is None:
                    raise FileNotFoundError(f"Image {member_name} missing in {archive_path}")
                data = member.read()

        suffix = Path(member_name).suffix or ".png"
        safe_name = hashlib.sha256(image_ref.encode("utf-8")).hexdigest() + suffix
        target = self._temp_path / safe_name
        with open(target, "wb") as f:
            f.write(data)

        self._cache[image_ref] = target
        return target


def _extract_blueprint_from_prompt(prompt: str) -> Optional[Dict[str, Any]]:
    """从 prompt 中提取 blueprint JSON 数据。"""
    marker = "Generate 3D objects for the following room blueprint:\n"
    if marker not in prompt:
        return None
    try:
        json_str = prompt.split(marker, 1)[1].strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return None


def _count_images_in_sequence(input_ids: List[int], tokenizer) -> int:
    """通过统计 <|vision_start|> 后紧跟 image_token 的数量来计算图片数。"""
    # 获取特殊 token ID
    vision_start_token_id = getattr(tokenizer, "vision_start_token_id", None)
    if vision_start_token_id is None:
        # 尝试从 vocab 中查找
        vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    
    image_token_id = getattr(tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    
    # 如果找不到这些 token，返回 0
    if vision_start_token_id is None or image_token_id is None:
        return 0
    
    # 统计 vision_start 后紧跟 image_token 的数量
    num_images = 0
    for i, token_id in enumerate(input_ids):
        if token_id == vision_start_token_id:
            # 检查下一个 token 是否是 image_token
            if i + 1 < len(input_ids) and input_ids[i + 1] == image_token_id:
                num_images += 1
    return num_images


def load_records_from_arrow(
    arrow_dir: str,
    tokenizer_name: str,
    trust_remote_code: bool,
    mode: str,
    logger,
    arrow_split: Optional[str] = None,
    arrow_offset: int = 0,
    arrow_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Decode user prompts from a packed arrow dataset produced by prepare_data.sh."""

    dataset_path = Path(arrow_dir).expanduser().resolve()
    dataset_obj = load_from_disk(str(dataset_path))
    if isinstance(dataset_obj, DatasetDict):
        split_name = arrow_split or "train"
        if split_name not in dataset_obj:
            raise ValueError(f"Split '{split_name}' not found in dataset at {dataset_path}")
        dataset: Dataset = dataset_obj[split_name]
    else:
        dataset = dataset_obj  # type: ignore[assignment]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        use_fast=True
    )

    has_sequence_lengths = "sequence_lengths" in dataset.column_names
    has_has_image = "has_image" in dataset.column_names
    has_image_refs = "image_refs" in dataset.column_names

    image_extractor: Optional[ArrowImageExtractor] = None
    if has_image_refs:
        image_extractor = ArrowImageExtractor(dataset_path)

    records: List[Dict[str, Any]] = []
    skipped = 0
    total_sequences = 0

    for row_idx in range(len(dataset)):
        row = dataset[row_idx]
        seq_lengths = row.get("sequence_lengths") if has_sequence_lengths else None
        if not seq_lengths:
            seq_lengths = [len(row["input_ids"])]
        offsets = list(itertools.accumulate([0] + seq_lengths))
        has_image_flags = row.get("has_image") if has_has_image else None
        if not has_image_flags:
            has_image_flags = [False] * len(seq_lengths)
        image_refs = row.get("image_refs") if has_image_refs else None
        if not image_refs:
            image_refs = []

        # 用于追踪展平后的 image_refs 索引
        global_image_ref_idx = 0

        for seq_idx, seq_len in enumerate(seq_lengths):
            start = offsets[seq_idx]
            end = offsets[seq_idx + 1]
            input_ids = row["input_ids"][start:end]
            labels = row["labels"][start:end]

            # 通过解析 input_ids 计算该子序列的图片数量
            num_images = _count_images_in_sequence(input_ids, tokenizer)

            prompt = _decode_user_prompt(tokenizer, input_ids, labels)
            if not prompt:
                # 即使跳过该序列，也要更新 image_ref 索引
                global_image_ref_idx += num_images
                continue

            total_sequences += 1
            if skipped < arrow_offset:
                skipped += 1
                global_image_ref_idx += num_images
                continue

            # 默认使用 arrow 索引作为文件名
            default_filename = f"arrow_{row_idx:06d}_seg{seq_idx:02d}"
            record = {
                "messages": [{"role": "user", "content": prompt}],
                "filename": default_filename,
                "__source_path": f"{dataset_path}#row={row_idx}/seg={seq_idx}",
            }

            # blueprint 模式：从 prompt 中提取 blueprint 数据
            if mode == "blueprint":
                blueprint_data = _extract_blueprint_from_prompt(prompt)
                if blueprint_data:
                    record["blueprint"] = blueprint_data

            # panorama/multi_view 模式：提取 ground truth
            if mode in ("panorama", "multi_view") and num_images > 0:
                gt = _decode_assistant_response(tokenizer, input_ids, labels)
                if gt:
                    record["ground_truth"] = gt

            if has_image_refs and has_image_flags[seq_idx] and num_images > 0:
                if image_extractor is None:
                    raise ValueError("VL samples detected but image archives are unavailable")
                
                # 从展平的 image_refs 中提取该子序列对应的图片引用
                seq_image_refs = image_refs[global_image_ref_idx:global_image_ref_idx + num_images]
                
                # 完全根据用户指定的 mode 参数决定处理方式
                if mode == "multi_view" and seq_image_refs:
                    # multi_view 模式：统一使用 image_paths 列表（无论图片数量）
                    image_paths = []
                    for ref in seq_image_refs:
                        if ref:
                            image_paths.append(str(image_extractor.materialize(ref)))
                    if image_paths:
                        record["image_paths"] = image_paths
                        # 使用第一张图片的文件名作为样本名
                        if seq_image_refs[0] and "::" in seq_image_refs[0]:
                            _, member_name = seq_image_refs[0].split("::", 1)
                            # 提取父目录名作为样本标识
                            record["filename"] = Path(member_name).parent.name or Path(member_name).stem
                elif mode == "panorama" and seq_image_refs:
                    # panorama 模式：统一使用 image_path 单张图片（无论图片数量，取第一张）
                    image_ref = seq_image_refs[0]
                    if image_ref:
                        record["image_path"] = str(image_extractor.materialize(image_ref))
                        # 从 image_ref 中提取原始图像文件名 (格式: archive_rel::member_name)
                        if "::" in image_ref:
                            _, member_name = image_ref.split("::", 1)
                            record["filename"] = Path(member_name).stem

            # 更新全局 image_ref 索引
            global_image_ref_idx += num_images
            records.append(record)

            if arrow_count is not None and len(records) >= arrow_count:
                logger.info(
                    "Loaded %d sequences from %s (processed %d rows)",
                    len(records),
                    dataset_path,
                    row_idx + 1
                )
                return records

    logger.info(
        "Loaded %d sequences from %s (total rows=%d, skipped=%d, mode=%s)",
        len(records),
        dataset_path,
        len(dataset),
        skipped,
        mode,
    )
    return records


class VLLMInferenceEngine:
    """Thin wrapper around vLLM for chat-style inference."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str],
        adapter_path: Optional[str],
        dtype: str,
        max_model_len: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        trust_remote_code: bool,
        image_resolution: Optional[str] = None,
        max_num_seqs: Optional[int] = None,
    ):
        self.logger = setup_logger("vllm_inference")
        # 解析图像分辨率参数
        self.image_resolution = None
        if image_resolution:
            try:
                w, h = image_resolution.lower().split("x")
                self.image_resolution = (int(w), int(h))
                self.logger.info("Image resolution set to: %dx%d", self.image_resolution[0], self.image_resolution[1])
            except ValueError:
                self.logger.warning("Invalid image_resolution format '%s', expected 'WIDTHxHEIGHT' (e.g., '1024x512'). Using original resolution.", image_resolution)
        tokenizer_target = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_target,
            trust_remote_code=trust_remote_code,
            use_fast=True
        )

        llm_kwargs: Dict[str, Any] = {
            "model": model_path,
            "tokenizer": tokenizer_target,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "disable_log_stats": True,  # 禁用 vLLM 内部统计日志
        }
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs

        self.adapter_path = adapter_path
        self.adapter_rank = None

        if adapter_path:
            self.adapter_rank = self._detect_lora_rank(adapter_path)
            llm_kwargs.update({
                "enable_lora": True,
                "max_lora_rank": self.adapter_rank,
                "max_loras": 1,
                "max_cpu_loras": 1,
            })
            self.logger.info("Initializing vLLM with LoRA support (rank=%s)", self.adapter_rank)
        else:
            self.logger.info("Loading vLLM with full model: %s", model_path)

        self.llm = LLM(**llm_kwargs)

        if self.adapter_path:
            self.logger.info("LoRA adapter configured: %s (will be applied during generation)", self.adapter_path)

    def _detect_lora_rank(self, adapter_path: str) -> int:
        config_path = Path(adapter_path) / "adapter_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return int(cfg.get("r", cfg.get("lora_r", 64)))
        return 64

    def build_prompt(self, messages: List[Dict[str, Any]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate(self, samples: Sequence[SampleInput], sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        payloads: List[Any] = []
        pending_samples: List[SampleInput] = []
        pending_indices: List[int] = []
        results: List[Optional[Dict[str, Any]]] = [None] * len(samples)

        for idx, sample in enumerate(samples):
            prompt_text = self.build_prompt(sample.messages)
            entry: Dict[str, Any] = {"prompt": prompt_text}

            # 优先处理 multi_view 模式（多图片）
            image_list: List[Image.Image] = []
            if sample.image_data_list:
                image_list = list(sample.image_data_list)
            elif sample.image_paths:
                try:
                    for img_path in sample.image_paths:
                        image_list.append(Image.open(img_path).convert("RGB"))
                except Exception as exc:
                    results[idx] = {
                        "sample": sample,
                        "text": "",
                        "tokens": 0,
                        "prompt_tokens": 0,
                        "error": f"image_load_error: {exc}"
                    }
                    continue
            elif sample.image_data is not None:
                image_list = [sample.image_data]
            elif sample.image_path:
                try:
                    image_list = [Image.open(sample.image_path).convert("RGB")]
                except Exception as exc:
                    results[idx] = {
                        "sample": sample,
                        "text": "",
                        "tokens": 0,
                        "prompt_tokens": 0,
                        "error": f"image_load_error: {exc}"
                    }
                    continue

            if image_list:
                # 如果设置了图像分辨率，则 resize 所有图像
                if self.image_resolution:
                    image_list = [img.resize(self.image_resolution, Image.LANCZOS) for img in image_list]
                entry["multi_modal_data"] = {"image": image_list}

            payloads.append(entry)
            pending_samples.append(sample)
            pending_indices.append(idx)

        generated = []
        if payloads:
            lora_request = None
            if self.adapter_path:
                lora_request = LoRARequest("default", 1, self.adapter_path)
            generated = self.llm.generate(payloads, sampling_params, lora_request=lora_request, use_tqdm=False)

        for idx, sample, output in zip(pending_indices, pending_samples, generated):
            if not output.outputs:
                results[idx] = {
                    "sample": sample,
                    "text": "",
                    "tokens": 0,
                    "prompt_tokens": len(output.prompt_token_ids or []),
                    "error": "empty_output"
                }
                continue

            completion = output.outputs[0]
            results[idx] = {
                "sample": sample,
                "text": completion.text,
                "tokens": len(completion.token_ids),
                "prompt_tokens": len(output.prompt_token_ids or []),
                "error": None
            }

        return [res for res in results if res is not None]


def load_records(
    input_path: Optional[str],
    single_prompt: Optional[str],
    image_path: Optional[str],
    image_dir: Optional[str],
    mode: str,
) -> List[Dict[str, Any]]:
    if not any([input_path, single_prompt, image_path, image_dir]):
        raise ValueError("Provide at least --input_path, --prompt, --image_path, or --image_dir")

    records: List[Dict[str, Any]] = []

    if input_path:
        target = Path(input_path).expanduser().resolve()
        if target.is_dir():
            files = sorted(target.glob("*.json")) + sorted(target.glob("*.jsonl"))
            if not files:
                raise ValueError(f"No JSON/JSONL files found in directory: {target}")
            for file_path in files:
                records.extend(_load_file(file_path))
        else:
            records.extend(_load_file(target))
    elif single_prompt is not None and not (image_path or image_dir):
        records.append({"prompt": single_prompt})

    image_records: List[Dict[str, Any]] = []
    if image_path or image_dir:
        if mode not in ("panorama", "multi_view"):
            raise ValueError("--image_path/--image_dir can only be used when --mode panorama or multi_view")
        prompt_for_images = single_prompt
        if image_path:
            image_records.extend(_build_image_records([Path(image_path)], prompt_for_images))
        if image_dir:
            dir_path = Path(image_dir).expanduser().resolve()
            if not dir_path.exists():
                raise FileNotFoundError(f"Image directory not found: {dir_path}")
            
            # multi_view 模式：将每个子文件夹作为一个样本，文件夹内所有图片组成 image_paths
            if mode == "multi_view":
                subdirs = sorted([d for d in dir_path.iterdir() if d.is_dir()])
                if not subdirs:
                    raise ValueError(f"No subdirectories found under directory: {dir_path}")
                for subdir in subdirs:
                    images = sorted([
                        str(p) for p in subdir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                    ])
                    if images:
                        record: Dict[str, Any] = {
                            "id": subdir.name,
                            "image_paths": images,
                            "__source_dir": str(subdir),
                            "__source_path": str(subdir),
                        }
                        if prompt_for_images:
                            record["prompt"] = prompt_for_images
                        image_records.append(record)
                if not image_records:
                    raise ValueError(f"No image files found in subdirectories under: {dir_path}")
            else:
                # panorama 模式：原有逻辑，将每张图片作为一个样本
                image_files = [
                    p for p in sorted(dir_path.rglob("*"))
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
                if not image_files:
                    raise ValueError(f"No image files found under directory: {dir_path}")
                image_records.extend(_build_image_records(image_files, prompt_for_images))
        records.extend(image_records)

    if not records:
        raise ValueError("No valid input records were constructed")

    for idx, rec in enumerate(records):
        rec["__row_id"] = idx
    return records


def _load_file(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".txt":
        return _load_txt(path)
    raise ValueError(f"Unsupported file type: {path}")


def _attach_source(record: Dict[str, Any], path: Path) -> Dict[str, Any]:
    record["__source_path"] = str(path)
    record["__source_dir"] = str(path.parent)
    return record


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [_attach_source(dict(item), path) for item in data]
    if isinstance(data, dict):
        return [_attach_source(dict(data), path)]
    raise ValueError(f"JSON file must contain object or list: {path}")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(_attach_source(json.loads(line), path))
    return records


def _load_txt(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            prompt = line.strip()
            if not prompt:
                continue
            records.append(_attach_source({"prompt": prompt}, path))
    return records


def _build_image_records(paths: List[Path], prompt: Optional[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw_path in paths:
        resolved = raw_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Image file not found: {resolved}")
        record: Dict[str, Any] = {
            "image_path": str(resolved),
            "filename": resolved.stem,
            "__source_path": str(resolved),
            "__source_dir": str(resolved.parent),
        }
        if prompt:
            record["prompt"] = prompt
        records.append(record)
    return records


def save_json(output_dir: Path, sample_id: str, data: Any, compact: bool, blueprint: Optional[Dict[str, Any]] = None) -> Path:
    filename = f"{sample_id}.json"
    target = output_dir / filename
    # 如果有 blueprint 条件，放到 JSON 开头
    if blueprint and isinstance(data, dict):
        data = {"blueprint": blueprint, **data}
    with open(target, "w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return target


def save_condition_asset(condition_dir: Path, sample: SampleInput, logger) -> None:
    condition_dir.mkdir(exist_ok=True)
    cond_type = sample.condition_type

    if cond_type == "text":
        text_content = (sample.condition_value or sample.prompt_preview or "").strip()
        if not text_content:
            logger.warning("Skipping text condition save for %s due to empty content", sample.uid)
            return
        target = condition_dir / f"{sample.uid}.txt"
        with open(target, "w", encoding="utf-8") as f:
            f.write(text_content + "\n")
        return

    if cond_type == "blueprint":
        blueprint = sample.condition_value
        if not isinstance(blueprint, dict):
            logger.warning("Blueprint condition missing or invalid for %s", sample.uid)
            return
        target = condition_dir / f"{sample.uid}.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump(blueprint, f, ensure_ascii=False, indent=2)
        return

    if cond_type == "panorama":
        target = condition_dir / f"{sample.uid}.png"
        image_obj: Optional[Image.Image] = None
        if sample.image_data is not None:
            try:
                image_obj = sample.image_data.copy()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Unable to clone in-memory panorama image for %s: %s", sample.uid, exc)
                return
        elif sample.image_path is not None:
            try:
                with Image.open(sample.image_path) as img:
                    image_obj = img.convert("RGB")
            except Exception as exc:
                logger.warning("Unable to load panorama image for %s: %s", sample.uid, exc)
                return
        if image_obj is None:
            logger.warning("No panorama data available for %s", sample.uid)
            return
        image_obj.save(target, format="PNG")
        image_obj.close()
        return

    if cond_type == "multi_view":
        # 收集所有图片
        images: List[Image.Image] = []
        
        # 优先使用内存中的图片数据
        if sample.image_data_list:
            for img in sample.image_data_list:
                try:
                    images.append(img.copy().convert("RGB"))
                except Exception as exc:
                    logger.warning("Unable to load multi-view image for %s: %s", sample.uid, exc)
        # 从文件路径加载
        elif sample.image_paths:
            for img_path in sample.image_paths:
                try:
                    with Image.open(img_path) as img:
                        images.append(img.convert("RGB").copy())
                except Exception as exc:
                    logger.warning("Unable to load multi-view image %s for %s: %s", img_path, sample.uid, exc)
        
        if not images:
            logger.warning("No multi-view images available for %s", sample.uid)
            return
        
        # 将多张图片拼接成网格
        import math
        n = len(images)
        # 计算网格布局：尽量接近正方形
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        # 统一所有图片尺寸（使用第一张图片的尺寸）
        cell_w, cell_h = images[0].size
        
        # 创建网格画布
        grid_w = cols * cell_w
        grid_h = rows * cell_h
        grid_img = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
        
        # 将图片粘贴到网格中
        for i, img in enumerate(images):
            row_idx = i // cols
            col_idx = i % cols
            x = col_idx * cell_w
            y = row_idx * cell_h
            # 如果图片尺寸不一致，resize 到统一尺寸
            if img.size != (cell_w, cell_h):
                img = img.resize((cell_w, cell_h), Image.LANCZOS)
            grid_img.paste(img, (x, y))
        
        # 保存网格图片
        target = condition_dir / f"{sample.uid}.png"
        grid_img.save(target, format="PNG")
        
        # 清理
        for img in images:
            img.close()
        grid_img.close()
        return

    logger.debug("Condition type '%s' is not supported for saving (sample=%s)", cond_type, sample.uid)


def run_text_interactive_mode(args):
    """交互式text模式：循环等待用户输入，生成后继续等待下一个"""
    import random
    
    engine = VLLMInferenceEngine(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        adapter_path=args.adapter_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        image_resolution=args.image_resolution,
        max_num_seqs=args.max_num_seqs,
    )
    
    # 保存基础采样参数（不包含 seed），每次生成时动态创建带新 seed 的 SamplingParams
    base_sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "stop": args.stop,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "n": 1,
    }
    builder = MessageBuilder("caption", image_root=args.image_root)

    # 与其它模式一致：创建时间戳子目录
    base_output_dir = Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "json"
    condition_dir = output_dir / "conditions"
    failed_dir = output_dir / "failed"
    json_dir.mkdir(exist_ok=True)
    condition_dir.mkdir(exist_ok=True)
    failed_dir.mkdir(exist_ok=True)

    print(f"\n=== Text Mode (type 'quit' to exit) ===")
    print(f"Output: {output_dir}")
    print(f"Note: Random seed is regenerated for each prompt to ensure varied outputs.")
    idx = 0
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        if not user_input or user_input.lower() == "quit":
            print("Exiting...")
            break

        uid = f"text_{idx:04d}"
        sample = builder.build({"prompt": user_input})
        
        # 每次生成时使用新的随机种子，确保相同 prompt 也能得到不同结果
        current_seed = random.randint(0, 2**31 - 1)
        sampling_params = SamplingParams(
            **base_sampling_kwargs,
            seed=current_seed,
        )
        print(f"[Using random seed: {current_seed}]")
        
        results = engine.generate([sample], sampling_params)
        raw_text = results[0]["text"]
        parsed_obj, json_text, _ = parse_json_from_text(raw_text)

        # 保存条件（prompt）
        with open(condition_dir / f"{uid}.txt", "w", encoding="utf-8") as f:
            f.write(user_input)

        if parsed_obj is not None:
            save_json(json_dir, uid, parsed_obj, compact=args.compact_json)
            print(f"[Saved to json/{uid}.json]")
        else:
            with open(failed_dir / f"{uid}.txt", "w", encoding="utf-8") as f:
                f.write(raw_text)
            print(f"[JSON parse failed, saved to failed/{uid}.txt]")
        idx += 1


def run_cli():
    parser = argparse.ArgumentParser(description="vLLM inference for JSON scene generation")
    parser.add_argument("--model_path", required=True, help="Path or name of the (fine-tuned) base model")
    parser.add_argument("--adapter_path", help="Path to LoRA adapter directory")
    parser.add_argument("--tokenizer_path", help="Optional tokenizer path (defaults to model_path)")
    parser.add_argument("--mode", default="unconditional", choices=["unconditional", "caption", "blueprint", "panorama", "multi_view", "text"], help="Prompt construction mode")
    parser.add_argument("--input_path", help="Directory / file containing raw inputs")
    parser.add_argument("--prompt", help="Single prompt string (overrides input_path)")
    parser.add_argument("--image_path", help="Single image for panorama mode")
    parser.add_argument("--image_dir", help="Directory containing images for panorama mode")
    parser.add_argument("--arrow_dir", help="Path to a tokenized arrow dataset (save_to_disk) for generating prompts")
    parser.add_argument("--arrow_split", help="Split name when arrow_dir contains a DatasetDict (default: train)")
    parser.add_argument("--arrow_offset", type=int, default=0, help="Number of packed sequences to skip when reading arrow_dir")
    parser.add_argument("--arrow_count", type=int, help="Maximum number of sequences to decode from arrow_dir")
    parser.add_argument("--image_root", help="Root directory for relative panorama image paths")
    parser.add_argument("--image_resolution", default="1024x512", help="Input image resolution in WIDTHxHEIGHT format (default: 1024x512)")
    parser.add_argument("--output_dir", required=True, help="Directory to store outputs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for vLLM requests")
    parser.add_argument("--max_model_len", type=int, default=16384, help="Maximum model context length")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum new tokens per sample")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--stop", nargs="*", default=None, help="Optional stop sequences")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_num_seqs", type=int, default=None, help="Maximum number of sequences per iteration (lower to reduce memory)")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--compact_json", action="store_true", help="Save extracted JSON without indentation")
    parser.add_argument("--skip_json_files", action="store_true", help="Skip writing per-sample JSON files")
    parser.add_argument("--eval_mode", default="3d_scene", choices=["3d_scene", "clothes"], help="Evaluation script to use (default: 3d_scene)")
    parser.add_argument("--log_file", help="Optional path for log file")

    args = parser.parse_args()

    logger = setup_logger("vllm_inference", log_file=args.log_file)
    tokenizer_target = args.tokenizer_path or args.model_path

    records: List[Dict[str, Any]] = []
    if args.arrow_dir:
        arrow_records = load_records_from_arrow(
            arrow_dir=args.arrow_dir,
            tokenizer_name=tokenizer_target,
            trust_remote_code=args.trust_remote_code,
            mode=args.mode,
            logger=logger,
            arrow_split=args.arrow_split,
            arrow_offset=max(0, args.arrow_offset),
            arrow_count=args.arrow_count,
        )
        records.extend(arrow_records)

    manual_inputs = any([args.input_path, args.prompt, args.image_path, args.image_dir])
    if manual_inputs:
        records.extend(load_records(
            input_path=args.input_path,
            single_prompt=args.prompt,
            image_path=args.image_path,
            image_dir=args.image_dir,
            mode=args.mode,
        ))

    # text模式：交互式循环输入
    if args.mode == "text":
        run_text_interactive_mode(args)
        return

    if not records:
        raise ValueError("Provide --arrow_dir or at least one of --input_path/--prompt/--image_path/--image_dir")

    if args.limit is not None:
        records = records[: args.limit]
    logger.info("Loaded %d input records", len(records))

    builder = MessageBuilder(args.mode, image_root=args.image_root)
    samples: List[SampleInput] = []
    for record in records:
        try:
            samples.append(builder.build(record))
        except ValueError as exc:
            logger.warning("Skipping record due to validation error: %s", exc)
    logger.info("Prepared %d samples after validation", len(samples))

    if not samples:
        logger.error("No valid samples to process")
        return

    engine = VLLMInferenceEngine(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        adapter_path=args.adapter_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        image_resolution=args.image_resolution,
        max_num_seqs=args.max_num_seqs,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=args.stop,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        seed=args.seed,
        n=1,
    )

    # 自动在 output_dir 下创建日期时间子目录
    base_output_dir = Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    json_dir = output_dir / "json"
    failed_dir = output_dir / "failed"
    condition_dir = output_dir / "conditions"
    ground_truth_dir = output_dir / "ground_truth"
    if not args.skip_json_files:
        json_dir.mkdir(exist_ok=True)
    failed_dir.mkdir(exist_ok=True)
    condition_dir.mkdir(exist_ok=True)
    # panorama/multi_view 模式下创建 ground_truth 目录
    if args.mode in ("panorama", "multi_view") and args.arrow_dir:
        ground_truth_dir.mkdir(exist_ok=True)

    responses_file = output_dir / "responses.jsonl"
    stats = {
        "total": len(samples),
        "successful": 0,
        "failed": 0,
        "json_parsed": 0,
        "json_failed": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "mode": args.mode,
        "adapter": bool(args.adapter_path),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 多线程写文件相关的辅助函数
    def _save_ground_truth(gt_dir: Path, uid: str, gt_content: str):
        """保存 ground truth JSON 文件"""
        gt_path = gt_dir / f"{uid}.json"
        try:
            gt_obj = json.loads(gt_content)
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt_obj, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            with open(gt_path, "w", encoding="utf-8") as f:
                f.write(gt_content)
        # 可视化 blueprint 多边形
        _visualize_blueprint(gt_content, gt_dir / f"{uid}.png")

    def _save_prediction(json_dir: Path, uid: str, parsed_obj: Any, compact: bool, bp: Optional[Dict], mode: str):
        """保存预测结果 JSON 文件"""
        save_json(json_dir, uid, parsed_obj, compact=compact, blueprint=bp)
        # panorama/multi_view 模式下可视化生成结果的 blueprint
        if mode in ("panorama", "multi_view"):
            _visualize_blueprint(parsed_obj, json_dir / f"{uid}.png")

    def _save_failed(failed_dir: Path, uid: str, raw_text: str):
        """保存解析失败的原始输出"""
        failure_path = failed_dir / f"{uid}.txt"
        with open(failure_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

    start = time.time()
    # 创建线程池用于异步写文件（使用 8 个工作线程）
    write_executor = ThreadPoolExecutor(max_workers=32)
    write_futures: List[Future] = []
    # 用于保护 responses.jsonl 的线程锁
    responses_lock = threading.Lock()
    # 用于收集所有 output_record，最后统一写入
    all_output_records: List[str] = []

    with open(responses_file, "w", encoding="utf-8") as writer:
        batch_size = max(1, args.batch_size)
        total_batches = (len(samples) + batch_size - 1) // batch_size
        # 禁用 tqdm 自带的速度显示，改用自定义的速度统计
        pbar = tqdm(total=len(samples), desc="Generating", unit="sample", 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]")
        total_tokens = 0  # 累计 token 数
        processed_samples = 0  # 累计处理的样本数
        for batch in chunked(samples, batch_size):
            batch_start_time = time.time()
            batch_results = engine.generate(batch, sampling_params)
            batch_tokens = 0
            for result in batch_results:
                sample: SampleInput = result["sample"]
                raw_text = result["text"]
                prompt_tokens = result["prompt_tokens"]
                completion_tokens = result["tokens"]
                stats["prompt_tokens"] += prompt_tokens
                stats["completion_tokens"] += completion_tokens
                batch_tokens += completion_tokens

                parsed_obj, json_text, parse_error = parse_json_from_text(raw_text)
                output_record = {
                    "id": sample.uid,
                    "prompt": sample.prompt_preview,
                    "raw_response": raw_text,
                    "json_text": json_text,
                    "json_error": parse_error,
                    "metadata": sample.metadata,
                }

                # 异步保存条件资源
                write_futures.append(
                    write_executor.submit(save_condition_asset, condition_dir, sample, logger)
                )

                # 异步保存 ground truth（仅 panorama/multi_view 模式且从 arrow 加载时）
                if sample.ground_truth and ground_truth_dir.exists():
                    gt_content = sample.ground_truth
                    # 去掉 "assistant\n" 前缀，只保留 JSON 部分
                    if gt_content.startswith("assistant\n"):
                        gt_content = gt_content[len("assistant\n"):]
                    write_futures.append(
                        write_executor.submit(_save_ground_truth, ground_truth_dir, sample.uid, gt_content)
                    )

                if parsed_obj is not None:
                    stats["json_parsed"] += 1
                    if not args.skip_json_files:
                        # blueprint 模式下将条件放到 JSON 开头
                        bp = sample.condition_value if sample.condition_type == "blueprint" else None
                        # 异步保存预测结果
                        write_futures.append(
                            write_executor.submit(_save_prediction, json_dir, sample.uid, parsed_obj, args.compact_json, bp, args.mode)
                        )
                else:
                    stats["json_failed"] += 1
                    # 异步保存失败结果
                    write_futures.append(
                        write_executor.submit(_save_failed, failed_dir, sample.uid, raw_text)
                    )

                # 收集 output_record，稍后统一写入（避免频繁加锁）
                all_output_records.append(json.dumps(output_record, ensure_ascii=False) + "\n")
                stats["successful" if parsed_obj is not None else "failed"] += 1
            # 批次处理完成后统一更新进度条
            batch_elapsed = time.time() - batch_start_time
            total_tokens += batch_tokens
            processed_samples += len(batch_results)
            pbar.update(len(batch_results))
            # 计算速度：使用总耗时计算准确的速度
            total_elapsed = time.time() - start
            if total_elapsed > 0:
                samples_per_sec = processed_samples / total_elapsed
                tokens_per_sec = total_tokens / total_elapsed
                pbar.set_postfix({
                    "s/sample": f"{total_elapsed/processed_samples:.2f}",
                    "tok/s": f"{tokens_per_sec:.1f}"
                })
        pbar.close()

        # 统一写入 responses.jsonl（在主线程中完成，避免线程安全问题）
        for record_line in all_output_records:
            writer.write(record_line)

    # 等待所有异步写文件任务完成
    logger.info("Waiting for %d async write tasks to complete...", len(write_futures))
    for future in write_futures:
        try:
            future.result()  # 等待完成，并捕获异常
        except Exception as e:
            logger.warning("Async write task failed: %s", e)
    write_executor.shutdown(wait=True)
    logger.info("All async write tasks completed.")

    stats["elapsed_seconds"] = round(time.time() - start, 2)
    with open(output_dir / "generation_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Inference completed in %.2f seconds", stats["elapsed_seconds"])
    logger.info(
        "Parsed %d/%d samples into JSON (%.2f%%)",
        stats["json_parsed"],
        stats["total"],
        100 * stats["json_parsed"] / max(1, stats["total"])
    )

    # 自动调用评估脚本（仅 panorama/multi_view 模式且有 ground_truth）
    if args.mode in ("panorama", "multi_view") and ground_truth_dir.exists():
        gt_files = list(ground_truth_dir.glob("*.json"))
        if gt_files and json_dir.exists():
            current_script = Path(__file__).resolve()
            eval_dir = current_script.parent.parent.parent / "eval"
            
            if args.eval_mode == "clothes":
                logger.info("Running clothes metrics evaluation...")
                eval_script = eval_dir / "clothes_compute_metrics.py"
                eval_cmd = ["python", str(eval_script), "--base_dir", str(output_dir), "--output", str(output_dir)+"/result.json"]
            else:  # 3d_scene
                logger.info("Running F1-score evaluation...")
                eval_script = eval_dir / "f1_score.py"
                eval_cmd = ["python", str(eval_script), "--pred_dir", str(json_dir), "--gt_dir", str(ground_truth_dir)]
            
            if eval_script.exists():
                try:
                    result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        logger.info("Evaluation completed successfully")
                        for line in result.stdout.strip().split("\n")[-10:]:
                            logger.info(line)
                    else:
                        logger.warning("Evaluation failed with return code %d", result.returncode)
                        if result.stderr:
                            logger.warning("stderr: %s", result.stderr[:500])
                except subprocess.TimeoutExpired:
                    logger.warning("Evaluation timed out (600s)")
                except Exception as e:
                    logger.warning("Failed to run evaluation: %s", e)
            else:
                logger.warning("Eval script not found at %s", eval_script)
        else:
            logger.info("Skipping evaluation: no ground truth files or prediction files")


if __name__ == "__main__":
    run_cli()
