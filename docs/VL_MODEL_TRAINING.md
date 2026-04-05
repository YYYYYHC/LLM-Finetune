# 视觉语言模型 (VL) 训练支持

本文档描述了为支持 Qwen-VL 等视觉语言模型训练所做的改动和设计思路。

## 目录

- [功能概述](#功能概述)
- [设计思路](#设计思路)
- [文件改动](#文件改动)
- [使用方法](#使用方法)
- [数据格式](#数据格式)
- [注意事项](#注意事项)

---

## 功能概述

新增 **panorama 模式**，支持使用全景图像作为条件输入，训练 VL 模型生成 3D 场景 JSON。

主要特性：
- 支持 Qwen2-VL、Qwen2.5-VL 等视觉语言模型的 LoRA 微调
- 图像数据存储到 Arrow 格式，支持高效加载
- 支持混合数据训练（纯文本 + 带图像数据）
- 自动检测 VL 模型并使用相应的 processor

---

## 设计思路

### 1. 数据准备阶段

**目标**：将全景图像和对应的 JSON 数据打包存储到 HuggingFace Dataset 格式中。

**设计决策**：
- 使用 HuggingFace 的 `Image` feature 存储图像，Arrow 格式会自动压缩存储
- 在数据准备阶段进行 tokenization，但保留原始图像供训练时处理
- VL 模型使用 `AutoProcessor` 而非 `AutoTokenizer`

**数据流**：
```
JSON文件 + 全景图像
      ↓
load_panorama_json_pairs()  # 加载配对数据
      ↓
convert_to_qwen_format(mode="panorama")  # 转换为对话格式
      ↓
tokenize_vl_function()  # VL专用tokenization
      ↓
Arrow Dataset (含 input_ids, labels, image)
```

### 2. 模型加载阶段

**目标**：自动识别 VL 模型并使用正确的加载方式。

**设计决策**：
- 通过模型名称自动检测是否为 VL 模型（包含 "VL", "Vision" 等关键词）
- VL 模型使用 `Qwen2VLForConditionalGeneration` 加载
- LoRA 只应用于 LLM 部分，不修改视觉编码器

**模型层级**：
```
Qwen2-VL Model
├── Visual Encoder (冻结)
└── Language Model
    └── LoRA Adapters (可训练)
        ├── q_proj, k_proj, v_proj, o_proj
        └── gate_proj, up_proj, down_proj
```

### 3. 训练阶段

**目标**：在训练时正确处理图像数据并传递给模型。

**设计决策**：
- 在 `collate_fn` 中动态处理图像，生成 `pixel_values`
- 支持混合 batch（部分样本有图像，部分没有）
- 使用 processor 的 `image_processor` 处理图像

**Collate 流程**：
```
Batch of samples
      ↓
提取 input_ids, labels
      ↓
动态 padding 到 batch 最大长度
      ↓
如果有图像：使用 image_processor 处理
      ↓
返回 {input_ids, attention_mask, labels, pixel_values, ...}
```

---

## 文件改动

### 1. `src/data/prepare_dataset.py`

**新增函数**：

| 函数 | 说明 |
|------|------|
| `load_panorama_json_pairs()` | 加载全景图像与 JSON 的配对数据 |
| `tokenize_vl_function()` | VL 模型专用的 tokenization 函数 |

**修改函数**：

| 函数 | 改动 |
|------|------|
| `convert_to_qwen_format()` | 新增 `mode="panorama"` 支持 |
| `prepare_dataset_simple()` | 新增 `image_dir` 参数，支持 VL 数据处理 |
| `prepare_dataset_batch()` | 新增 `image_dir` 参数，支持 VL 数据处理 |
| `prepare_dataset()` | 新增 `image_dir` 参数传递 |
| `main()` | 新增 `--image_dir` 和 `--mode panorama` 参数 |

**关键代码片段**：

```python
def load_panorama_json_pairs(json_dir: str, image_dir: str) -> List[Dict[str, Any]]:
    """
    加载配对的全景图像和JSON文件。
    图像路径结构: image_dir/{json_stem}/panorama/panorama_rgb.png
    """
    # ... 匹配 JSON 文件和对应的全景图像
    for stem, json_file in json_files.items():
        panorama_path = image_path / stem / "panorama" / "panorama_rgb.png"
        if panorama_path.exists():
            data.append({
                "image_path": str(panorama_path),
                "json_data": json_data,
                "filename": stem
            })
```

### 2. `src/models/model_factory.py`

**新增函数**：

| 函数 | 说明 |
|------|------|
| `is_vision_language_model()` | 检测模型是否为 VL 模型 |
| `load_processor()` | 加载 VL 模型的 processor |
| `load_vl_model_lora()` | 加载 VL 模型并应用 LoRA |

**修改函数**：

| 函数 | 改动 |
|------|------|
| `create_model()` | 新增 `is_vl_model` 参数，支持 VL 模型创建 |

**关键代码片段**：

```python
def load_vl_model_lora(model_name, lora_config, ...):
    """加载 VL 模型并应用 LoRA"""
    from transformers import Qwen2VLForConditionalGeneration
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype_val,
        attn_implementation="flash_attention_2"
    )
    
    # LoRA 只应用于 LLM 部分
    peft_config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", ...],
        ...
    )
    model = get_peft_model(model, peft_config)
```

### 3. `src/training/trainer.py`

**修改 `__init__`**：
- 自动检测 VL 模型
- VL 模型加载 processor 而非 tokenizer

**修改 `_collate_fn`**：
- 支持处理图像数据
- 生成 `pixel_values` 和 `image_grid_thw`

**关键代码片段**：

```python
def _collate_fn(self, batch):
    # ... 文本处理 ...
    
    # VL 模型图像处理
    if has_images and self.processor is not None:
        valid_images = [img for img in images if img is not None]
        if valid_images:
            image_inputs = self.processor.image_processor(
                images=valid_images,
                return_tensors="pt"
            )
            result["pixel_values"] = image_inputs["pixel_values"]
            if "image_grid_thw" in image_inputs:
                result["image_grid_thw"] = image_inputs["image_grid_thw"]
```

### 4. `scripts/prepare_data.sh`

**新增配置选项**：
```bash
# panorama 模式配置
MODE="panorama"
IMAGE_DIR="./data/demo_images"
```

**新增参数传递**：
```bash
--mode "$MODE" \
--image_dir "$IMAGE_DIR"
```

### 5. 新增文件

| 文件 | 说明 |
|------|------|
| `configs/lora_qwen_vl.yaml` | Qwen-VL LoRA 微调配置 |
| `scripts/prepare_data_panorama.sh` | 全景数据准备脚本 |
| `scripts/train_lora_vl.sh` | VL 模型训练脚本 |

---

## 使用方法

### 步骤 1: 准备数据

确保你的数据目录结构如下：

```
data/
├── SingleRoom_json_289k/           # JSON 文件目录
│   ├── seed_10121810_Bathroom.json
│   ├── seed_10122112_LivingRoom.json
│   └── ...
└── demo_images/                    # 图像目录
    ├── seed_10121810_Bathroom/
    │   └── panorama/
    │       └── panorama_rgb.png
    ├── seed_10122112_LivingRoom/
    │   └── panorama/
    │       └── panorama_rgb.png
    └── ...
```

修改并运行数据准备脚本：

```bash
# 编辑 scripts/prepare_data_panorama.sh
JSON_DIR="./data/SingleRoom_json_289k"
IMAGE_DIR="./data/demo_images"
OUTPUT_DIR="./data/SingleRoom_panorama_qwen_vl"
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"

# 运行
bash scripts/prepare_data_panorama.sh
```

或者直接使用命令行：

```bash
python -m src.data.prepare_dataset \
    --json_dir ./data/SingleRoom_json_289k \
    --output_dir ./data/SingleRoom_panorama_qwen_vl \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --mode panorama \
    --image_dir ./data/demo_images \
    --max_length 4096 \
    --batch_mode \
    --batch_size 1000
```

### 步骤 2: 配置训练

编辑 `configs/lora_qwen_vl.yaml`：

```yaml
model_name: "Qwen/Qwen2-VL-7B-Instruct"
training_mode: "lora"
is_vl_model: true

batch_dirs:
  - base_dir: "./data/SingleRoom_panorama_qwen_vl"
    train_batches: "1-20"
    eval_batches: [0]

max_length: 4096
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

lora_config:
  r: 64
  lora_alpha: 16
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

### 步骤 3: 开始训练

```bash
# 单卡训练
python train.py --config configs/lora_qwen_vl.yaml

# 多卡训练
accelerate launch --config_file configs/accelerate_config.yaml \
    train.py --config configs/lora_qwen_vl.yaml
```

---

## 数据格式

### 输入数据格式

**JSON 文件** (`seed_xxx.json`)：
```json
{
    "blueprint": { ... },
    "object_1": { ... },
    "object_2": { ... }
}
```

**全景图像**：
- 路径: `{image_dir}/{json_filename}/panorama/panorama_rgb.png`
- 格式: PNG
- 建议分辨率: 按模型要求（Qwen-VL 支持动态分辨率）

### 处理后的数据集

数据集包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `input_ids` | List[int] | tokenized 输入序列 |
| `labels` | List[int] | 训练标签（user 部分为 -100） |
| `image` | Image | 全景图像（HuggingFace Image feature） |
| `has_image` | bool | 是否包含图像 |

---

## Packing 机制

### 原理说明

本项目的 packing 功能利用了 **Flash Attention 2** 的 `flash_attn_varlen_func` 特性：

1. **传统 padding 的问题**：不同长度的序列需要 pad 到相同长度，造成计算浪费
2. **Packing 解决方案**：将多个短序列拼接成一个长序列，通过 `position_ids` 标识子序列边界

**关键机制**：
- 通过 `position_ids` 从 0 重新开始来标识新的子序列
- 不传 `attention_mask`，让 transformers 自动检测 packed sequences
- Flash Attention 根据 position_ids 中的重置点调用 `flash_attn_varlen_func` 实现子序列间的注意力隔离

```
传统 padding:
[seq1_tokens][pad][pad][pad]  →  浪费计算
[seq2_tokens][pad]
[seq3_tokens][pad][pad]

Packing:
[seq1_tokens][seq2_tokens][seq3_tokens]  →  高效利用
position_ids: [0,1,2,3][0,1][0,1,2]  →  子序列边界标识
```

### VL 模型的 M-RoPE

Qwen2-VL/Qwen3-VL 使用 **M-RoPE (Multimodal RoPE)**，`position_ids` 是 3D 的：

```python
# position_ids 形状: (batch_size, seq_len, 3)
# 三个维度: (temporal, height, width)

# 对于纯文本 token，三个维度值相同:
position_ids[i, j] = [pos, pos, pos]

# 对于图像 token，需要 2D 网格位置:
position_ids[i, j] = [temporal, h_pos, w_pos]
```

### 使用方法

在数据准备时启用 packing：

```bash
# scripts/prepare_data.sh
PACKING="--packing"
```

训练时，`_collate_fn` 会自动：
1. 检测 `sequence_lengths` 字段判断是否为 packed 数据
2. 生成正确的 `position_ids`（普通 LLM 为 1D，VL 模型为 3D）
3. 不传 `attention_mask`，让 Flash Attention 处理子序列隔离

### 注意事项

- **VL + 图像 packing**：由于图像 token 的 position_ids 需要 2D 网格位置，带图像的数据 packing 较复杂
- **纯文本 packing**：VL 模型的纯文本数据可以正常使用 packing
- **效率提升**：packing 可显著减少 padding 浪费，提高训练效率

---

## 注意事项

### 显存要求

VL 模型比纯文本模型需要更多显存：

| 配置 | 估计显存 |
|------|---------|
| Qwen2-VL-7B + LoRA (bf16) | ~40GB per GPU |
| Qwen2-VL-7B + LoRA + 4bit | ~20GB per GPU |

**建议**：
- 使用 `per_device_train_batch_size: 1`
- 增大 `gradient_accumulation_steps` 来模拟更大 batch
- 启用 `gradient_checkpointing: true`

### 图像处理

- 图像在 `collate_fn` 中动态处理，不在数据准备阶段预处理
- 这样可以保持灵活性，支持不同的图像增强策略
- `dataloader_num_workers` 建议设为较小值（2-4）避免内存问题

### 混合数据训练

支持同时训练纯文本和带图像的数据：

```yaml
batch_dirs:
  - base_dir: "./data/text_only_data"      # 纯文本数据
    train_batches: "1-10"
  - base_dir: "./data/panorama_data"       # 带图像数据
    train_batches: "1-20"
```

训练时，collate_fn 会自动处理两种类型的数据。

### 模型兼容性

已测试支持的模型：
- `Qwen/Qwen2-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`

其他 VL 模型可能需要调整 `load_vl_model_lora()` 中的模型类。

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据准备流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  JSON文件 ─────┐                                                │
│                ├──→ load_panorama_json_pairs() ──→ 配对数据     │
│  全景图像 ─────┘                                                │
│                                                                 │
│  配对数据 ──→ convert_to_qwen_format(mode="panorama")          │
│           ──→ tokenize_vl_function()                           │
│           ──→ Arrow Dataset                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        训练流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Arrow Dataset                                                  │
│       ↓                                                         │
│  DataLoader + _collate_fn()                                    │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Batch:                                               │       │
│  │   - input_ids      (padded)                         │       │
│  │   - attention_mask                                   │       │
│  │   - labels         (user部分=-100)                  │       │
│  │   - pixel_values   (图像特征)                        │       │
│  │   - image_grid_thw (可选)                           │       │
│  └─────────────────────────────────────────────────────┘       │
│       ↓                                                         │
│  Qwen2-VL Model                                                │
│  ├── Visual Encoder ──→ 图像特征                               │
│  └── Language Model + LoRA ──→ 生成 JSON                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

