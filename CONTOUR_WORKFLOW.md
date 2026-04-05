# Contour Data LoRA Fine-tuning Workflow

Reproduce the full pipeline: B-spline contour JSON data -> tokenized Arrow dataset -> LoRA training -> inference.

Hardware tested: single NVIDIA RTX 4090 (24GB).

## 1. Environment Setup

```bash
conda create -n llm_finetune python=3.11 -y
conda activate llm_finetune

# PyTorch with CUDA (adjust cu128 to match your CUDA version)
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Project dependencies
pip install "transformers>=4.36.0" "accelerate>=0.25.0" "peft>=0.7.0" \
    "datasets>=2.16.0" "pyyaml>=6.0" "tqdm>=4.65.0" "wandb>=0.16.0" \
    "bitsandbytes>=0.41.0" "safetensors>=0.4.0" "sentencepiece>=0.1.99" \
    tensorboard

# Flash Attention (recommended, takes a few minutes to build)
pip install flash-attn --no-build-isolation
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## 2. Prepare Data

### 2a. Flatten nested directories

The data loader (`src/data/loaders.py`) only does non-recursive `glob("*.json")`, so nested directories must be flattened first.

```bash
python scripts/prepare_contour_data.py \
    --input_dir /home/yhc/cross_sdf/data/thick_structures_slices \
    --output_dir /tmp/contour_flat
```

This copies 17 JSON files (skipping metadata files `processed.json`, `categories_done.json`) into one flat directory.

### 2b. Tokenize

```bash
python -m src.data.prepare_dataset \
    --json_dir /apdcephfs_nj7/share_1220751/hcyuan/mesh_d8/slice_30uniform \
    --output_dir /apdcephfs_nj7/share_1220751/hcyuan/mesh_d8/slice_30uniform_token \
    --model_name "Qwen/Qwen3-8B" \
    --max_length 16384 \
    --mode contour \
    --seed 42 \
    --num_proc 64 \
    --test_split 0.05
```

Samples exceeding `max_length` are automatically dropped. With 8500, 3 out of 17 files survive (7681, 8344, 8456 tokens). The rest are too large.

To keep more samples, increase `--max_length` — but you'll need a GPU with more VRAM or a larger multi-GPU setup (see [GPU Memory Notes](#gpu-memory-notes) below).

Verify:

```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('./data/tokenized/contour_test')
for i, s in enumerate(ds['train']):
    print(f'Sample {i}: {len(s[\"input_ids\"])} tokens')
"
```

## 3. Train

### 3a. Launch training

```bash
conda activate llm_finetune

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
    --config_file configs/accelerate_config_single_gpu.yaml \
    train.py --config configs/lora_contour.yaml
```

With the default config (100 epochs, 3 samples), this runs 300 steps in ~5 minutes on a 4090.

### 3b. Key config knobs (in `configs/lora_contour.yaml`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model_name` | `Qwen/Qwen3-0.6B` | See GPU memory notes for larger models |
| `num_epochs` | 100 | 100 epochs x 3 samples = 300 steps |
| `learning_rate` | 2e-4 | Standard LoRA LR |
| `lora_config.r` | 32 | LoRA rank; increase for more capacity |
| `max_length` | 8500 | Must match tokenization max_length |
| `save_steps` | 50 | Checkpoint interval |
| `gradient_checkpointing` | true | Required for long sequences |
| `use_flash_attention` | true | Requires flash-attn package |

### 3c. Monitor

TensorBoard logs are saved to the output directory:

```bash
tensorboard --logdir ./outputs/contour_lora_qwen3_0.6b
```

### 3d. Output structure

```
outputs/contour_lora_qwen3_0.6b/
    checkpoint-50/          # Intermediate checkpoint
    checkpoint-100/
    ...
    final/                  # Final model
        adapter_config.json
        adapter_model.safetensors
        tokenizer.json
        tokenizer_config.json
    training_config.yaml
    training_summary.json
```

## 4. Inference

```bash
python scripts/inference_contour.py \
    --adapter_path ./outputs/contour_lora_qwen3_0.6b/final \
    --max_new_tokens 8192 \
    --temperature 0.3 \
    --output_file ./outputs/contour_lora_qwen3_0.6b/generated.json
```

Options:

| Flag | Default | Notes |
|------|---------|-------|
| `--base_model` | `Qwen/Qwen3-0.6B` | Must match training model |
| `--adapter_path` | (required) | Path to LoRA checkpoint |
| `--max_new_tokens` | 8192 | Max generation length |
| `--temperature` | 0.7 | Lower = more deterministic |
| `--top_p` | 0.9 | Nucleus sampling |
| `--output_file` | None | Save output to file |

Expected output: a valid JSON with the UDF contour token format:

```json
{
  "format": "udf_contour_tokens",
  "bounding_box_xz": [-0.258, 0.256, -0.256, 0.258],
  "slices": [
    {
      "y": -0.49,
      "contours": [
        {"closed": true, "deg": 3, "n": 5, "cx": [...], "cz": [...], "k": [...], "d": [...]}
      ]
    },
    ...
  ]
}
```

## 5. Quick-start (copy-paste)

All steps in one block:

```bash
conda activate llm_finetune

# Flatten
python scripts/prepare_contour_data.py \
    --input_dir /home/yhc/cross_sdf/data/thick_structures_slices \
    --output_dir /tmp/contour_flat

# Tokenize
python -m src.data.prepare_dataset \
    --json_dir /tmp/contour_flat \
    --output_dir ./data/tokenized/contour_test \
    --model_name "Qwen/Qwen3-0.6B" \
    --max_length 8500 \
    --mode contour \
    --seed 42 --num_proc 4 --test_split 0.0

# Train
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
    --config_file configs/accelerate_config_single_gpu.yaml \
    train.py --config configs/lora_contour.yaml

# Inference
python scripts/inference_contour.py \
    --adapter_path ./outputs/contour_lora_qwen3_0.6b/final \
    --max_new_tokens 8192 --temperature 0.3 \
    --output_file ./outputs/contour_lora_qwen3_0.6b/generated.json
```

## GPU Memory Notes

Qwen3's vocabulary is 151,936 tokens. The logits tensor during training is `(batch, seq_len, 151936)` in bf16, which dominates VRAM for long sequences:

| seq_len | Logits size (bf16) | Logits + grads |
|---------|--------------------|----------------|
| 4,096   | 1.2 GB             | ~2.4 GB        |
| 8,192   | 2.4 GB             | ~4.8 GB        |
| 16,384  | 4.7 GB             | ~9.4 GB        |
| 32,768  | 9.4 GB             | ~18.8 GB       |

Tested configurations on a single RTX 4090 (24GB):

| Model | Quantization | max_length | Result |
|-------|-------------|------------|--------|
| Qwen3-8B | 4-bit | 8,500 | OOM |
| Qwen3-1.7B | 4-bit | 8,500 | OOM |
| Qwen3-0.6B | none | 8,500 | Works (~10 GB used) |

To use larger models or longer sequences, you need either:
- Multi-GPU with tensor parallelism (splits the logits computation)
- A GPU with more VRAM (e.g., A100 80GB, H100)

## Files Modified/Created

| File | Change |
|------|--------|
| `scripts/prepare_contour_data.py` | New - flatten nested data dirs |
| `scripts/inference_contour.py` | New - lightweight inference script |
| `src/data/converters.py` | Added "contour" mode |
| `src/data/prepare_dataset.py` | Added "contour" to argparse choices |
| `src/training/trainer.py` | Fixed single-GPU crash (`torch.distributed.all_reduce` guard) |
| `configs/lora_contour.yaml` | New - training config |
| `configs/accelerate_config_single_gpu.yaml` | New - single-GPU accelerate config |
