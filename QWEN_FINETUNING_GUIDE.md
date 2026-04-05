# What You Need to Finetune Qwen with This Codebase

This project is a **production-grade LLM finetuning framework built specifically around Qwen models**. It supports Qwen2, Qwen2.5, Qwen3 (text-only) and Qwen2-VL, Qwen2.5-VL, Qwen3-VL (vision-language). Below is everything you need beyond the code itself.

---

## 1. Hardware Requirements

| Setup | VRAM per GPU | Notes |
|-------|-------------|-------|
| LoRA, text-only (Qwen 8B, 6K ctx) | ~20-30 GB | Single GPU possible |
| LoRA, VL model (Qwen-VL 7/8B) | ~40 GB | BF16 |
| LoRA + 4-bit quantization, VL | ~20 GB | QLoRA |
| Full finetune (Qwen 8B, 16K ctx) | 4-8x GPUs, 24 GB+ each | DeepSpeed ZeRO-3 required |

- **Recommended GPUs:** A100 (40/80 GB), H100, H20 (24 GB+)
- **Multi-node:** Supported via DeepSpeed ZeRO-3 with InfiniBand/NCCL

---

## 2. Software / Environment Setup

### Python Dependencies (`requirements.txt`)

```
torch==2.8.0
transformers>=4.36.0
accelerate>=0.25.0
peft>=0.7.0
datasets>=2.16.0
pyyaml>=6.0
sentencepiece>=0.1.99
safetensors>=0.4.0
bitsandbytes>=0.41.0
deepspeed
tensorboard
wandb>=0.16.0
torchvision
vllm==0.11.0          # inference only
```

### Additional System Requirements

- **CUDA 12.x** compatible driver
- **Flash Attention 2** — required for efficient training and sequence packing (`pip install flash-attn --no-build-isolation`)
- **GNU Parallel** (optional) — for parallel batch data preparation
- **HuggingFace Hub access** — to download Qwen model weights (may require `huggingface-cli login` for gated models)

---

## 3. Model Weights

Download from HuggingFace Hub. Common choices:

| Model | Type | HuggingFace ID |
|-------|------|----------------|
| Qwen3 8B | Text | `Qwen/Qwen3-8B` |
| Qwen2.5 8B | Text | `Qwen/Qwen2.5-8B` |
| Qwen3-VL 8B Instruct | Vision-Language | `Qwen/Qwen3-VL-8B-Instruct` |
| Qwen2.5-VL 7B Instruct | Vision-Language | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen2-VL 7B Instruct | Vision-Language | `Qwen/Qwen2-VL-7B-Instruct` |

Models auto-download on first use, or pre-download with:
```bash
huggingface-cli download Qwen/Qwen3-8B
```

---

## 4. Training Data

### Data Format

The framework expects **JSON files** as raw input. Each JSON represents a scene/task. The data pipeline converts them to Qwen's conversation format:

```json
{
  "messages": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User instruction"},
    {"role": "assistant", "content": "Expected output (e.g. JSON scene)"}
  ]
}
```

### Supported Input Modes

| Mode | Raw Input | Extra Data Needed |
|------|-----------|-------------------|
| `unconditional` | JSON files | Nothing |
| `blueprint` | JSON files | Nothing (blueprint subset used as input) |
| `caption` | JSON files + `.txt` caption files | A caption directory with matching filenames |
| `panorama` | JSON files + panorama images | Image directory: `{scene}/panorama/panorama_rgb.png` |
| `multi_view` | JSON files + orbit images | Image directory: `{scene}/orbit/orbit_*.png` |
| `multi_view2` | JSON files + flat images | Image directory: `{scene}/*.png` |
| `multi_view3` | JSON files + front view | Image directory: `{scene}/front.png` |

### Data Preparation

Run the preparation script to tokenize and batch-organize data into Arrow format:

```bash
# Edit scripts/prepare_data.sh with your paths, then:
bash scripts/prepare_data.sh
```

This produces numbered batch directories (`batch_0000/`, `batch_0001/`, ...) referenced in training configs.

---

## 5. Configuration Files You Must Create/Edit

### Training Config (YAML)

Copy and edit one of the provided templates:

| Template | Use Case |
|----------|----------|
| `configs/lora_qwen8b.yaml` | LoRA finetuning, text-only Qwen 8B |
| `configs/full_finetune_qwen8b.yaml` | Full finetuning, text-only Qwen 8B |
| `configs/lora_qwen_vl.yaml` | LoRA finetuning, Qwen-VL models |
| `configs/train.yaml` | Full finetuning, Qwen3-VL (currently active) |

**Key fields to set:**

```yaml
model_name: "Qwen/Qwen3-8B"          # or local path
training_mode: "lora"                  # or "full"

batch_dirs:
  - base_dir: "/path/to/your/tokenized/data"
    train_batches: "0-19"
    eval_batches: [20]

max_length: 6144                       # sequence length
output_dir: "./outputs/my_experiment"
```

### Accelerate / DeepSpeed Config

Pre-made configs in `configs/`:

| Config | Setup |
|--------|-------|
| `accelerate_config.yaml` | Simple multi-GPU |
| `accelerate_config_deepspeed_stage3.yaml` | DeepSpeed ZeRO-3 (recommended for full finetune) |
| `accelerate_config_4card.yaml` | 4-GPU |
| `accelerate_config_fsdp.yaml` | FSDP alternative |
| `accelerate_config_deepspeed_stage3_multi-nodes.yaml` | Multi-node |

---

## 6. Launch Training

```bash
# Single-node with DeepSpeed ZeRO-3
accelerate launch \
    --config_file configs/accelerate_config_deepspeed_stage3.yaml \
    train.py --config configs/lora_qwen8b.yaml

# Or use the wrapper script
bash train.sh

# Multi-node (run on each machine)
./train_multi_node.sh 0   # master
./train_multi_node.sh 1   # worker
```

---

## 7. Inference After Training

Uses **vLLM** for high-throughput inference:

```bash
# LoRA adapter inference
python -m src.inference.vllm_engine \
    --model_path Qwen/Qwen3-8B \
    --adapter_path ./outputs/my_experiment/final \
    --prompt "Your prompt here" \
    --output_dir ./results

# Full finetune checkpoint
python -m src.inference.vllm_engine \
    --model_path ./outputs/my_experiment/final \
    --output_dir ./results
```

---

## 8. Summary Checklist

| Item | Status |
|------|--------|
| GPUs with 24GB+ VRAM | Required |
| CUDA 12.x + Flash Attention 2 | Required |
| Python deps (`pip install -r requirements.txt`) | Required |
| Flash Attention (`pip install flash-attn`) | Required |
| Qwen model weights (auto-downloads or pre-download) | Required |
| Training data in JSON format | **You must provide** |
| Images (for VL modes: panorama/multi-view) | Required for VL only |
| Caption `.txt` files (for caption mode) | Required for caption mode only |
| Run `prepare_data.sh` to tokenize + batch data | Required before training |
| Edit a YAML config with your paths + hyperparams | Required |
| Choose an Accelerate config for your GPU setup | Required |
| W&B account (optional, for experiment tracking) | Optional |

**The codebase handles everything else** — model loading, tokenization, chat template formatting, LoRA injection, distributed training, checkpointing, and inference are all built in.
