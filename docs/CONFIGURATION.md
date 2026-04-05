# Configuration Guide

Complete guide to all configuration options in the fine-tuning framework.

## Configuration Files

- `configs/full_finetune.yaml`: Full fine-tuning configuration
- `configs/lora_finetune.yaml`: LoRA fine-tuning configuration
- `configs/accelerate_config.yaml`: Accelerate multi-GPU configuration

## Model Configuration

### `model_name`
- **Type**: String
- **Default**: `"Qwen/Qwen2.5-8B"`
- **Description**: HuggingFace model name or local path
- **Examples**:
  ```yaml
  model_name: "Qwen/Qwen2.5-8B"
  model_name: "Qwen/Qwen2.5-14B"
  model_name: "/path/to/local/model"
  ```

### `training_mode`
- **Type**: String
- **Options**: `"full"` or `"lora"`
- **Default**: Depends on config file
- **Description**: Training mode selection

### `torch_dtype`
- **Type**: String
- **Options**: `"float32"`, `"float16"`, `"bfloat16"`
- **Default**: `"bfloat16"`
- **Description**: Data type for model weights
- **Recommendation**: Use `"bfloat16"` for H20 GPUs

### `use_flash_attention`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable Flash Attention 2 for faster training
- **Note**: Requires compatible hardware

## Data Configuration

### `dataset_path`
- **Type**: String
- **Required**: Yes
- **Description**: Path to processed HuggingFace dataset
- **Example**: `"./data/processed"`

### `max_length`
- **Type**: Integer
- **Default**: `2048`
- **Description**: Maximum sequence length
- **Range**: 128 - 8192 (model dependent)

## Training Hyperparameters

### `num_epochs`
- **Type**: Integer
- **Default**: `3`
- **Description**: Number of training epochs
- **Recommendation**: 1-5 for most tasks

### `per_device_train_batch_size`
- **Type**: Integer
- **Default**: `2` (full), `4` (LoRA)
- **Description**: Batch size per GPU
- **Recommendation**: Adjust based on GPU memory

### `per_device_eval_batch_size`
- **Type**: Integer
- **Default**: `4` (full), `8` (LoRA)
- **Description**: Evaluation batch size per GPU

### `gradient_accumulation_steps`
- **Type**: Integer
- **Default**: `8` (full), `4` (LoRA)
- **Description**: Number of gradient accumulation steps
- **Note**: Effective batch size = `per_device_train_batch_size * num_gpus * gradient_accumulation_steps`

## Optimizer Configuration

### `optimizer`
- **Type**: String
- **Options**: `"adamw"`
- **Default**: `"adamw"`
- **Description**: Optimizer type

### `learning_rate`
- **Type**: Float
- **Default**: `2e-5` (full), `3e-4` (LoRA)
- **Description**: Learning rate
- **Recommendations**:
  - Full: 1e-5 to 5e-5
  - LoRA: 1e-4 to 5e-4

### `weight_decay`
- **Type**: Float
- **Default**: `0.01`
- **Description**: Weight decay for regularization

### `adam_betas`
- **Type**: List[Float]
- **Default**: `[0.9, 0.999]`
- **Description**: Adam beta parameters

### `adam_epsilon`
- **Type**: Float
- **Default**: `1e-8`
- **Description**: Adam epsilon for numerical stability

### `max_grad_norm`
- **Type**: Float
- **Default**: `1.0`
- **Description**: Maximum gradient norm for clipping

## Learning Rate Scheduler

### `lr_scheduler_type`
- **Type**: String
- **Options**: `"linear"`, `"cosine"`, `"polynomial"`, `"constant"`
- **Default**: `"cosine"`
- **Description**: Learning rate scheduler type

### `warmup_ratio`
- **Type**: Float
- **Default**: `0.1`
- **Description**: Ratio of total steps for warmup
- **Alternative**: Use `warmup_steps` for exact step count

### `warmup_steps`
- **Type**: Integer
- **Optional**: Yes
- **Description**: Exact number of warmup steps
- **Note**: If set, overrides `warmup_ratio`

## LoRA Configuration

### `lora_config.r`
- **Type**: Integer
- **Default**: `64`
- **Description**: LoRA rank (number of low-rank dimensions)
- **Recommendations**:
  - `8-16`: Very efficient, simple tasks
  - `32-64`: Balanced, most tasks
  - `128+`: High capacity

### `lora_config.lora_alpha`
- **Type**: Integer
- **Default**: `16`
- **Description**: LoRA scaling factor
- **Formula**: Effective scaling = `lora_alpha / r`

### `lora_config.lora_dropout`
- **Type**: Float
- **Default**: `0.1`
- **Description**: Dropout probability for LoRA layers

### `lora_config.target_modules`
- **Type**: List[String]
- **Default**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **Description**: Which modules to apply LoRA to
- **Options**: Any linear layer in the model

### `lora_config.bias`
- **Type**: String
- **Options**: `"none"`, `"all"`, `"lora_only"`
- **Default**: `"none"`
- **Description**: How to handle bias terms

### `load_in_4bit`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Load model in 4-bit quantization
- **Note**: Reduces memory, slight quality loss

### `load_in_8bit`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Load model in 8-bit quantization

## Distributed Training

### `mixed_precision`
- **Type**: String
- **Options**: `"no"`, `"fp16"`, `"bf16"`
- **Default**: `"bf16"`
- **Description**: Mixed precision training mode

### `gradient_checkpointing`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable gradient checkpointing for memory efficiency

## Logging and Saving

### `output_dir`
- **Type**: String
- **Required**: Yes
- **Description**: Directory to save model and checkpoints
- **Example**: `"./outputs/lora_finetune"`

### `logging_steps`
- **Type**: Integer
- **Default**: `10`
- **Description**: Log metrics every N steps

### `save_steps`
- **Type**: Integer
- **Default**: `500`
- **Description**: Save checkpoint every N steps

### `eval_steps`
- **Type**: Integer
- **Default**: `500`
- **Description**: Run evaluation every N steps

## Evaluation

### `do_eval`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Whether to run evaluation during training

## Other Settings

### `seed`
- **Type**: Integer
- **Default**: `42`
- **Description**: Random seed for reproducibility

### `dataloader_num_workers`
- **Type**: Integer
- **Default**: `4`
- **Description**: Number of workers for data loading

### `padding_side`
- **Type**: String
- **Options**: `"left"`, `"right"`
- **Default**: `"right"`
- **Description**: Where to add padding tokens

## Weights & Biases (Optional)

### `log_with`
- **Type**: String
- **Options**: `"wandb"`, `"tensorboard"`
- **Default**: `"tensorboard"`
- **Description**: Logging backend

### `wandb_project`
- **Type**: String
- **Optional**: Yes
- **Description**: W&B project name

### `wandb_run_name`
- **Type**: String
- **Optional**: Yes
- **Description**: W&B run name

## Example: Custom Configuration

```yaml
# Custom configuration example
model_name: "Qwen/Qwen2.5-14B"  # Use larger model
training_mode: "lora"

dataset_path: "./data/processed"
max_length: 4096  # Longer sequences

# Training
num_epochs: 5  # More epochs
per_device_train_batch_size: 2  # Smaller batch
gradient_accumulation_steps: 16  # Compensate with accumulation

# Optimizer
learning_rate: 2.0e-4
weight_decay: 0.05  # More regularization

# LoRA
lora_config:
  r: 128  # Higher rank
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
    - "lm_head"  # Include output layer

# Quantization for memory
load_in_4bit: true

# Scheduler
lr_scheduler_type: "cosine"
warmup_ratio: 0.05  # Less warmup

# Logging
output_dir: "./outputs/custom_experiment"
logging_steps: 5
save_steps: 250
eval_steps: 250

# W&B
log_with: "wandb"
wandb_project: "qwen-scene-generation"
wandb_run_name: "experiment-v1"

seed: 42
```

## Tips for Configuration

### Memory Optimization

If you run out of memory:

1. **Reduce batch size**:
   ```yaml
   per_device_train_batch_size: 1
   ```

2. **Increase gradient accumulation**:
   ```yaml
   gradient_accumulation_steps: 32
   ```

3. **Enable quantization** (LoRA only):
   ```yaml
   load_in_4bit: true
   ```

4. **Reduce sequence length**:
   ```yaml
   max_length: 1024
   ```

### Speed Optimization

For faster training:

1. **Enable Flash Attention**:
   ```yaml
   use_flash_attention: true
   ```

2. **Increase batch size** (if memory allows):
   ```yaml
   per_device_train_batch_size: 8
   ```

3. **Use LoRA** instead of full fine-tuning

4. **Reduce evaluation frequency**:
   ```yaml
   eval_steps: 1000
   ```

### Quality Optimization

For better model quality:

1. **Increase epochs**:
   ```yaml
   num_epochs: 5
   ```

2. **Use full fine-tuning** (if resources allow)

3. **Increase LoRA rank**:
   ```yaml
   lora_config:
     r: 128
   ```

4. **Lower learning rate**:
   ```yaml
   learning_rate: 1.0e-5
   ```

5. **More warmup**:
   ```yaml
   warmup_ratio: 0.2
   ```

## Validation

Validate your config before training:

```python
from src.utils.config_loader import load_config, validate_config

config = load_config("configs/my_config.yaml")
required_keys = ["model_name", "dataset_path", "output_dir", "training_mode"]
validate_config(config, required_keys)
```

## Best Practices

1. **Start small**: Test with a subset first
2. **Monitor closely**: Watch loss curves
3. **Save checkpoints**: Configure appropriate `save_steps`
4. **Version control**: Track config changes
5. **Document experiments**: Use descriptive names

## Need Help?

- Check example configs in `configs/`
- Run `python scripts/estimate_training_time.py` to estimate resources
- See `USAGE.md` for detailed usage guide

