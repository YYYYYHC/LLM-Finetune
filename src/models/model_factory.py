"""
Model factory for initializing models with different configurations.
Supports both text-only and vision-language models.
"""

from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("model_factory")


def is_vision_language_model(model_name: str) -> bool:
    """Check if the model is a vision-language model based on its name."""
    vl_indicators = ["vl", "vision", "VL", "Vision", "Qwen3-VL"]
    return any(indicator in model_name for indicator in vl_indicators)


def load_tokenizer(
    model_name: str,
    **kwargs
) -> AutoTokenizer:
    """
    Load tokenizer for the model.
    
    Args:
        model_name: HuggingFace model name or path
        **kwargs: Additional arguments for tokenizer
    
    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer from {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        **kwargs
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def load_processor(
    model_name: str,
    **kwargs
) -> AutoProcessor:
    """
    Load processor for vision-language models.
    
    Args:
        model_name: HuggingFace model name or path
        **kwargs: Additional arguments for processor
    
    Returns:
        Loaded processor
    """
    logger.info(f"Loading processor from {model_name}")
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        **kwargs
    )
    
    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    return processor


def load_model_full_finetune(
    model_name: str,
    torch_dtype: str = "bfloat16",
    use_flash_attention: bool = True,
    **kwargs
) -> AutoModelForCausalLM:
    """
    Load model for full fine-tuning.
    
    Args:
        model_name: HuggingFace model name or path
        torch_dtype: Torch dtype for model weights
        use_flash_attention: Whether to use flash attention 2
        **kwargs: Additional arguments for model loading
    
    Returns:
        Loaded model
    """
    logger.info(f"Loading model for full fine-tuning: {model_name}")
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype_val = dtype_map.get(torch_dtype, torch.bfloat16)
    
    # Model loading arguments
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype_val,
        "device_map": None,  # Let accelerate handle device mapping
        **kwargs
    }
    
    # Add flash attention if requested
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model


def load_model_lora(
    model_name: str,
    lora_config: Dict[str, Any],
    torch_dtype: str = "bfloat16",
    use_flash_attention: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    **kwargs
) -> AutoModelForCausalLM:
    """
    Load model with LoRA adapters.
    
    Args:
        model_name: HuggingFace model name or path
        lora_config: LoRA configuration dictionary
        torch_dtype: Torch dtype for model weights
        use_flash_attention: Whether to use flash attention 2
        load_in_4bit: Whether to load in 4-bit quantization
        load_in_8bit: Whether to load in 8-bit quantization
        **kwargs: Additional arguments for model loading
    
    Returns:
        Model with LoRA adapters
    """
    logger.info(f"Loading model with LoRA: {model_name}")
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype_val = dtype_map.get(torch_dtype, torch.bfloat16)
    
    # Quantization config
    quantization_config = None
    if load_in_4bit:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype_val,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif load_in_8bit:
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Model loading arguments
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype_val,
        **kwargs
    }
    
    # IMPORTANT: When using DeepSpeed/FSDP, we must not set device_map
    # Let accelerate handle device mapping. Also, quantization may not work
    # well with DeepSpeed, so we only apply it if explicitly requested.
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        # For quantization, we need device_map="auto" or specific mapping
        # But this conflicts with DeepSpeed! User should be warned.
        logger.warning(
            "Using quantization with DeepSpeed may cause issues. "
            "Consider using FSDP instead or disable quantization."
        )
    
    # Add flash attention if requested
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load base model
    # For DeepSpeed ZeRO-3, we should load on CPU first to avoid OOM
    logger.info("Loading model (will be moved to correct device by Accelerate/DeepSpeed)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    logger.info("Model loaded successfully")
    
    # Prepare model for k-bit training if quantized
    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    else:
        # Enable gradient checkpointing BEFORE applying LoRA (for non-quantized models)
        # This is crucial for memory efficiency with long sequences
        model.gradient_checkpointing_enable()
    
    # Create LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias=lora_config.get("bias", "none"),
        inference_mode=False
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Re-enable gradient checkpointing after applying LoRA
    # PEFT models need this to properly work with gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    # For FSDP compatibility, explicitly enable gradient checkpointing on the PEFT model
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Convert LoRA parameters to the same dtype as base model to avoid FSDP dtype mismatch
    # This is critical for FSDP which requires uniform dtype within each shard
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only convert trainable parameters (LoRA params)
            param.data = param.data.to(torch_dtype_val)
    
    logger.info(f"Converted all trainable parameters to {torch_dtype}")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def load_vl_model_lora(
    model_name: str,
    lora_config: Dict[str, Any],
    torch_dtype: str = "bfloat16",
    use_flash_attention: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    **kwargs
):
    """
    Load Vision-Language model (e.g., Qwen-VL) with LoRA adapters.
    
    Args:
        model_name: HuggingFace model name or path
        lora_config: LoRA configuration dictionary
        torch_dtype: Torch dtype for model weights
        use_flash_attention: Whether to use flash attention 2
        load_in_4bit: Whether to load in 4-bit quantization
        load_in_8bit: Whether to load in 8-bit quantization
        **kwargs: Additional arguments for model loading
    
    Returns:
        Model with LoRA adapters
    """
    from transformers import Qwen3VLForConditionalGeneration
    
    logger.info(f"Loading VL model with LoRA: {model_name}")
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype_val = dtype_map.get(torch_dtype, torch.bfloat16)
    
    # Quantization config
    quantization_config = None
    if load_in_4bit:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype_val,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif load_in_8bit:
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Model loading arguments
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype_val,
        **kwargs
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        logger.warning(
            "Using quantization with VL models. "
            "This may affect image processing quality."
        )
    
    # Add flash attention if requested
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load VL model (Qwen3-VL)
    logger.info("Loading Qwen3-VL model (will be moved to correct device by Accelerate)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs
    )
    logger.info("VL model loaded successfully")
    
    # Prepare model for k-bit training if quantized
    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()
    
    # Create LoRA config - for VL models, we typically target the LLM part
    # Default target modules for Qwen-VL LLM backbone
    default_vl_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        target_modules=lora_config.get("target_modules", default_vl_target_modules),
        bias=lora_config.get("bias", "none"),
        inference_mode=False
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Re-enable gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Convert LoRA parameters to the same dtype
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch_dtype_val)
    
    logger.info(f"Converted all trainable parameters to {torch_dtype}")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def load_vl_model_full_finetune(
    model_name: str,
    torch_dtype: str = "bfloat16",
    use_flash_attention: bool = True,
    freeze_vision_encoder: bool = True,
    freeze_vision_merger: bool = True,
    **kwargs
):
    """
    Load Vision-Language model (e.g., Qwen-VL) for full fine-tuning.
    
    Args:
        model_name: HuggingFace model name or path
        torch_dtype: Torch dtype for model weights
        use_flash_attention: Whether to use flash attention 2
        freeze_vision_encoder: Whether to freeze the vision encoder (default: True)
                              If True, only the LLM part will be trained, vision encoder weights are frozen.
        freeze_vision_merger: Whether to freeze the vision-language connector/merger (default: True)
                             If False, the merger layer (visual.merger) will be trained.
        **kwargs: Additional arguments for model loading
    
    Returns:
        Loaded VL model for full fine-tuning
    """
    from transformers import Qwen3VLForConditionalGeneration
    
    logger.info(f"Loading VL model for full fine-tuning: {model_name}")
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype_val = dtype_map.get(torch_dtype, torch.bfloat16)
    
    # Model loading arguments
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype_val,
        "device_map": None,  # Let accelerate handle device mapping
        **kwargs
    }
    
    # Add flash attention if requested
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load VL model (Qwen3-VL)
    logger.info("Loading Qwen3-VL model for full fine-tuning (will be moved to correct device by Accelerate)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs
    )
    logger.info("VL model loaded successfully for full fine-tuning")
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Freeze vision encoder and/or merger if requested
    # Qwen-VL 的 vision encoder 模块名包含 "visual"，merger 模块名为 "visual.merger"
    if freeze_vision_encoder or freeze_vision_merger:
        frozen_encoder_params = 0
        frozen_merger_params = 0
        trainable_merger_params = 0
        for name, param in model.named_parameters():
            if "visual" in name:
                # 判断是否是 merger 层（连接层）
                is_merger = "merger" in name
                
                if is_merger:
                    # Merger 层：根据 freeze_vision_merger 决定是否冻结
                    if freeze_vision_merger:
                        param.requires_grad = False
                        frozen_merger_params += param.numel()
                    else:
                        # 保持可训练
                        trainable_merger_params += param.numel()
                else:
                    # Vision encoder 层：根据 freeze_vision_encoder 决定是否冻结
                    if freeze_vision_encoder:
                        param.requires_grad = False
                        frozen_encoder_params += param.numel()
        
        if freeze_vision_encoder:
            logger.info(f"Frozen vision encoder parameters: {frozen_encoder_params:,}")
        if freeze_vision_merger:
            logger.info(f"Frozen vision merger parameters: {frozen_merger_params:,}")
        else:
            logger.info(f"Trainable vision merger parameters: {trainable_merger_params:,}")
        
        if freeze_vision_encoder and freeze_vision_merger:
            logger.info("Only LLM backbone will be trained, vision encoder and merger are frozen.")
        elif freeze_vision_encoder and not freeze_vision_merger:
            logger.info("LLM backbone and vision merger will be trained, vision encoder is frozen.")
        elif not freeze_vision_encoder and freeze_vision_merger:
            logger.info("LLM backbone and vision encoder will be trained, merger is frozen.")
    
    # Print total parameters info
    # 注意：DeepSpeed ZeRO-3 模式下，参数被分片，需要使用特殊方法统计
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        # 在 ZeRO-3 下，参数可能有 ds_numel 属性表示真实的参数数量
        # ds_numel 是 DeepSpeed 添加的属性，表示分片前的完整参数数量
        if hasattr(param, 'ds_numel'):
            num_params = param.ds_numel
        else:
            num_params = param.numel()
        
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    if total_params > 0:
        logger.info(f"Trainable%: {100 * trainable_params / total_params:.2f}%")
    else:
        # 如果仍然是0，可能是模型还未完全初始化
        logger.warning("Trainable%: N/A (parameters count is 0, model may not be fully initialized)")
    
    return model


def create_model(
    model_name: str,
    training_mode: str = "full",
    lora_config: Optional[Dict[str, Any]] = None,
    is_vl_model: Optional[bool] = None,
    **kwargs
):
    """
    Factory method to create model based on training mode.
    
    Args:
        model_name: HuggingFace model name or path
        training_mode: Training mode ("full" or "lora")
        lora_config: LoRA configuration (required if training_mode="lora")
        is_vl_model: Whether this is a vision-language model. If None, auto-detect.
        **kwargs: Additional arguments passed to model loading
    
    Returns:
        Initialized model
    
    Raises:
        ValueError: If invalid training mode or missing config
    """
    # Auto-detect if VL model
    if is_vl_model is None:
        is_vl_model = is_vision_language_model(model_name)
    
    # Extract freeze_vision_encoder and freeze_vision_merger from kwargs for VL full finetune
    freeze_vision_encoder = kwargs.pop("freeze_vision_encoder", True)
    freeze_vision_merger = kwargs.pop("freeze_vision_merger", True)
    
    if is_vl_model:
        logger.info(f"Detected Vision-Language model: {model_name}")
        if training_mode == "lora":
            if lora_config is None:
                raise ValueError("lora_config is required for LoRA training")
            return load_vl_model_lora(model_name, lora_config, **kwargs)
        elif training_mode == "full":
            return load_vl_model_full_finetune(
                model_name, 
                freeze_vision_encoder=freeze_vision_encoder, 
                freeze_vision_merger=freeze_vision_merger,
                **kwargs
            )
        else:
            raise ValueError(f"Invalid training mode: {training_mode}")
    else:
        if training_mode == "full":
            return load_model_full_finetune(model_name, **kwargs)
        elif training_mode == "lora":
            if lora_config is None:
                raise ValueError("lora_config is required for LoRA training")
            return load_model_lora(model_name, lora_config, **kwargs)
        else:
            raise ValueError(f"Invalid training mode: {training_mode}")

