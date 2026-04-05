# Execution Log: Contour Data LoRA Fine-tuning Pipeline
Date: 2026-04-02

## Step 1: Create Conda Environment
- Created conda env `llm_finetune` with Python 3.11
- Installed PyTorch 2.8.0 with CUDA 12.8: `pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128`
- Installed remaining dependencies: transformers, accelerate, peft, datasets, wandb, bitsandbytes, safetensors, sentencepiece, tensorboard
- Installed flash-attn 2.8.3: `pip install flash-attn --no-build-isolation`
- Verified: PyTorch 2.8.0+cu128, CUDA available, RTX 4090 detected

## Step 2: Create Data Flatten Script
- Created `scripts/prepare_contour_data.py`
- Walks `/home/yhc/cross_sdf/data/thick_structures_slices/` recursively
- Skips metadata files (processed.json, categories_done.json)
- Copies 17 JSON data files into a flat output directory

## Step 3: Add "contour" Mode to Pipeline
- Modified `src/data/converters.py`:
  - Added "contour" elif branch after "unconditional" (lines ~128-145)
  - Updated docstring to include "contour" mode
  - Updated error message to list "contour" in supported modes
  - Contour format: user prompt "Generate B-spline contour slices..." + assistant = compact JSON
- Modified `src/data/prepare_dataset.py`:
  - Added "contour" to argparse choices (line 1023)

## Step 4: Run Data Preparation Pipeline

### 4a: Flatten contour data
```
python scripts/prepare_contour_data.py \
    --input_dir /home/yhc/cross_sdf/data/thick_structures_slices \
    --output_dir /tmp/contour_flat
```
Result: 17 files copied to /tmp/contour_flat

### 4b: Tokenize (multiple attempts with different max_length)

**Attempt 1: max_length=8192 (plan's original suggestion)**
- Result: Only 1 sample survived (7681 tokens). Plan's token estimates were too low.

**Attempt 2: max_length=32768**
- Result: 10 samples survived (7681 to 32354 tokens). But caused OOM during training.

**Attempt 3: max_length=16384**
- Result: 5 samples survived (7681 to 14114 tokens). Still caused OOM during training.

**Attempt 4 (final): max_length=8500 with Qwen3-0.6B tokenizer**
```
python -m src.data.prepare_dataset \
    --json_dir /tmp/contour_flat \
    --output_dir ./data/tokenized/contour_test \
    --model_name "Qwen/Qwen3-0.6B" \
    --max_length 8500 \
    --mode contour \
    --seed 42 --num_proc 4 --test_split 0.0
```
Result: 3 samples survived (7681, 8344, 8456 tokens)
- 14 samples dropped due to exceeding max_length

## Step 5: Create Training Configuration
- Created `configs/lora_contour.yaml` for single RTX 4090
- Created `configs/accelerate_config_single_gpu.yaml` for single-GPU training

### Key configuration decisions:
- **Model**: Qwen3-0.6B (not 8B or 1.7B) — Qwen3's 151936 vocab size creates massive logits tensors (seq_len x 151936) that OOM on 24GB GPU for long sequences
- **No quantization**: 0.6B model is small enough (~1.2GB in bf16) to not need 4-bit
- **LoRA**: r=32, alpha=16, dropout=0.05, targeting all attention + MLP projections
- **Gradient checkpointing**: enabled to reduce activation memory
- **Flash attention**: enabled for efficiency
- **batch_size=1, gradient_accumulation=1**: memory constrained

## Step 6: Launch LoRA Training

### Training attempt timeline:

**Attempt 1: Qwen3-8B, 4-bit, max_length=32768**
- Error: FlashAttention2 not installed
- Fix: `pip install flash-attn --no-build-isolation`

**Attempt 2: Qwen3-8B, 4-bit, max_length=32768 (with flash-attn)**
- Error: OOM at lm_head (logits = 32K x 152K = 9.3GB)

**Attempt 3: Qwen3-8B, 4-bit, max_length=16384**
- Error: OOM at cross_entropy loss (logits = 14K x 152K = 4GB + model/activations)

**Attempt 4: Qwen3-1.7B, 4-bit, max_length=8500**
- Error: OOM during backward pass (18.5GB allocated + 4.7GB needed)

**Attempt 5: Qwen3-0.6B, no quant, max_length=8500**
- Step 1 succeeded (loss=0.1896) but crashed on `torch.distributed.all_reduce` in single-GPU mode
- Fix: Added guard `if torch.distributed.is_initialized()` in `src/training/trainer.py` line 1233

**Attempt 6 (successful): Qwen3-0.6B, no quant, max_length=8500**
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
accelerate launch --config_file configs/accelerate_config_single_gpu.yaml \
    train.py --config configs/lora_contour.yaml
```
- Training completed: 3 epochs x 3 batches = 9 steps
- Loss progression: 0.1896 -> 0.4601 -> 0.5356 -> 0.1297 -> 0.5120 -> 0.4076 -> 0.4011 -> 0.1148 -> 0.4927
- Checkpoint saved at step 5 and final model saved
- LoRA adapter: ~40MB (adapter_model.safetensors)
- Trainable params: 20,185,088 / 616,235,008 total (3.28%)

## Step 7: Bug Fix Applied
- Fixed `src/training/trainer.py` line 1233: wrapped `torch.distributed.all_reduce` in `if torch.distributed.is_initialized()` to support single-GPU training without a distributed process group.

## Output Files
- Checkpoint: `outputs/contour_lora_qwen3_0.6b/checkpoint-5/`
- Final model: `outputs/contour_lora_qwen3_0.6b/final/`
- TensorBoard logs: `outputs/contour_lora_qwen3_0.6b/contour_lora_qwen3_0.6b/`
- Training config: `outputs/contour_lora_qwen3_0.6b/training_config.yaml`
- Training summary: `outputs/contour_lora_qwen3_0.6b/training_summary.json`

## Step 8: Extended Training (100 epochs, 300 steps)
- Updated `configs/lora_contour.yaml`: num_epochs=100, save_steps=50
- Command:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
accelerate launch --config_file configs/accelerate_config_single_gpu.yaml \
    train.py --config configs/lora_contour.yaml
```
- 100 epochs x 3 batches = 300 steps, ~5 minutes total
- Loss progression:
  - Step 1: 0.1896 (start)
  - Step 50: converged to ~0.001 range
  - Step 300: 0.0012 (final)
- Checkpoints saved at steps 50, 100, 150, 200, 250, 300, and final

## Step 9: Inference Test
- Created `scripts/inference_contour.py` (lightweight transformers+peft inference, no vLLM)
- Command:
```
python scripts/inference_contour.py \
    --adapter_path ./outputs/contour_lora_qwen3_0.6b/final \
    --max_new_tokens 8192 --temperature 0.3
```
- Result: **Valid JSON output** (7653 tokens generated)
  - format: udf_contour_tokens
  - bounding_box_xz: [-0.258, 0.256, -0.256, 0.258]
  - 50 slices (y: -0.49 to 0.49)
  - Correct contour structure: closed=True, deg=3, keys=[closed, deg, n, cx, cz, k, d]
  - JSON size: 12,020 chars (comparable to cube.json reference at 13,024 chars)
- Note: Some middle slices have 0 contours (slice[25] empty), while reference has 1 contour per slice — the model learned the format well but content varies

## Key Lessons
1. The plan's token count estimates were ~2x too low for the actual Qwen3 tokenizer
2. Qwen3's 151936 vocabulary creates a memory bottleneck at the logits/loss computation (seq_len x vocab_size x dtype_bytes)
3. On a single RTX 4090 (24GB), even Qwen3-1.7B with 4-bit quantization cannot handle 8K+ token sequences due to the logits memory requirement
4. Qwen3-0.6B without quantization fits comfortably for 8.5K token sequences
5. The trainer had a distributed-only bug (torch.distributed.all_reduce without init_process_group check)
