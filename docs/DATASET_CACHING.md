# Dataset Caching Guide / 数据集缓存指南

## Overview / 概述

This project **fully supports** caching datasets in HuggingFace Dataset format for reuse across training sessions and machines.

本项目**完全支持**将数据集缓存为 HuggingFace Dataset 格式，以便在多次训练和跨机器使用。

## How It Works / 工作原理

### 1. Data Preparation Phase / 数据准备阶段

The `prepare_dataset.py` script:
1. Loads and converts JSON files
2. Tokenizes using Qwen tokenizer
3. **Saves as HuggingFace Dataset** to specified directory
4. Includes tokenizer config for portability

`prepare_dataset.py` 脚本：
1. 加载和转换 JSON 文件
2. 使用 Qwen tokenizer 进行分词
3. **保存为 HuggingFace Dataset** 到指定目录
4. 包含 tokenizer 配置以便移植

### 2. Training Phase / 训练阶段

The `trainer.py` loads the cached dataset directly:
- No re-tokenization needed
- Fast loading from disk
- Works across different machines

`trainer.py` 直接加载缓存的数据集：
- 无需重新分词
- 快速从磁盘加载
- 跨机器使用

## Configuration / 配置

### Dataset Path / 数据集路径

Your specified path: `/Users/lutaojiang/Desktop/project/InfiniGen-NPR/data/infinigen_yaml_226k`

This path is now configured in:
- `scripts/prepare_data.sh`
- `configs/lora_finetune.yaml`
- `configs/full_finetune.yaml`

已在以下文件中配置：
- `scripts/prepare_data.sh` - 数据准备脚本
- `configs/lora_finetune.yaml` - LoRA 训练配置
- `configs/full_finetune.yaml` - 全量训练配置

## Usage / 使用方法

### Option 1: Using Scripts / 使用脚本

```bash
# Prepare dataset (once)
bash scripts/prepare_data.sh
```

This will create the dataset at:
```
/Users/lutaojiang/Desktop/project/InfiniGen-NPR/data/infinigen_yaml_226k/
├── train/
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── test/
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── dataset_dict.json
└── tokenizer_config/
    └── (tokenizer files)
```

### Option 2: Manual Command / 手动命令

```bash
python -m src.data.prepare_dataset \
    --json_dir ./data/json \
    --output_dir /Users/lutaojiang/Desktop/project/InfiniGen-NPR/data/infinigen_yaml_226k \
    --model_name Qwen/Qwen2.5-8B \
    --max_length 2048 \
    --test_split 0.05
```

### Training with Cached Dataset / 使用缓存数据集训练

Once prepared, simply run training:

```bash
# The config files already point to the cached dataset
bash scripts/train_lora.sh
# or
bash scripts/train_full.sh
```

训练脚本会自动使用缓存的数据集。

## Dataset Format / 数据集格式

The cached dataset is in **HuggingFace Arrow format**:

```python
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk("/Users/lutaojiang/Desktop/project/InfiniGen-NPR/data/infinigen_yaml_226k")

# Structure
dataset
# DatasetDict({
#     train: Dataset({
#         features: ['input_ids', 'attention_mask', 'labels'],
#         num_rows: 190000
#     }),
#     test: Dataset({
#         features: ['input_ids', 'attention_mask', 'labels'],
#         num_rows: 10000
#     })
# })
```

## Advantages / 优势

### 1. No Re-tokenization / 无需重新分词
- Tokenization is done **once** during preparation
- Training starts immediately
- 分词只需进行**一次**
- 训练立即开始

### 2. Portable / 可移植
- Transfer the cached dataset to any machine
- No need for tokenizer download on training server
- Consistent preprocessing
- 可以传输到任何机器
- 训练服务器无需下载 tokenizer
- 预处理一致

### 3. Fast Loading / 快速加载
- Arrow format is memory-mapped
- Efficient for large datasets
- Arrow 格式支持内存映射
- 对大数据集高效

### 4. Version Control / 版本控制
- Cache different preprocessing versions
- Easy to compare results
- 缓存不同的预处理版本
- 便于比较结果

## Cross-Machine Workflow / 跨机器工作流程

### On Local Mac / 在本地 Mac 上

```bash
# 1. Convert YAML to JSON
bash scripts/convert_yaml_to_json.sh

# 2. Prepare and cache dataset
bash scripts/prepare_data.sh

# 3. Package for transfer
cd /Users/lutaojiang/Desktop/project/InfiniGen-NPR/data
tar -czf infinigen_yaml_226k.tar.gz infinigen_yaml_226k/
```

### Transfer / 传输

```bash
# Copy to GPU server
scp infinigen_yaml_226k.tar.gz user@gpu-server:/path/to/data/

# Or use rsync
rsync -avz infinigen_yaml_226k/ user@gpu-server:/path/to/data/infinigen_yaml_226k/
```

### On GPU Server / 在 GPU 服务器上

```bash
# 1. Extract (if compressed)
tar -xzf infinigen_yaml_226k.tar.gz

# 2. Update config to point to the dataset path
# Edit configs/lora_finetune.yaml or full_finetune.yaml
# Set: dataset_path: "/path/to/infinigen_yaml_226k"

# 3. Start training directly
bash scripts/train_lora.sh
```

## Verifying Cached Dataset / 验证缓存数据集

Check the dataset before training:

```python
from datasets import load_from_disk

# Load
dataset = load_from_disk("/Users/lutaojiang/Desktop/project/InfiniGen-NPR/data/infinigen_yaml_226k")

# Check sizes
print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# Check a sample
sample = dataset['train'][0]
print(f"Input IDs shape: {len(sample['input_ids'])}")
print(f"Attention mask shape: {len(sample['attention_mask'])}")
print(f"Labels shape: {len(sample['labels'])}")

# Verify tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/Users/lutaojiang/Desktop/project/InfiniGen-NPR/data/infinigen_yaml_226k/tokenizer_config"
)
decoded = tokenizer.decode(sample['input_ids'])
print(decoded[:200])  # Print first 200 chars
```

## Multiple Dataset Versions / 多个数据集版本

You can maintain multiple cached versions:

```bash
# Different max_length
python -m src.data.prepare_dataset \
    --output_dir /path/to/infinigen_yaml_226k_len1024 \
    --max_length 1024

# Different split
python -m src.data.prepare_dataset \
    --output_dir /path/to/infinigen_yaml_226k_split10 \
    --test_split 0.10
```

Then switch by updating `dataset_path` in config.

## Storage Requirements / 存储需求

Estimated sizes for 200k samples:

| Component | Size |
|-----------|------|
| Raw YAML files | ~100-150GB |
| JSON files | ~100-150GB |
| Cached HF Dataset | ~50-100GB |
| Compressed (tar.gz) | ~30-50GB |

缓存数据集比原始文件更紧凑！

## Best Practices / 最佳实践

### 1. Prepare Once, Use Many Times / 一次准备，多次使用
- Cache the dataset after initial preparation
- Reuse for multiple training runs
- 初始准备后缓存数据集
- 多次训练重复使用

### 2. Version Control / 版本控制
- Use descriptive names for different versions
- Document preprocessing parameters
- 为不同版本使用描述性名称
- 记录预处理参数

### 3. Backup / 备份
- Keep backup of cached dataset
- Cheaper than re-processing
- 保留缓存数据集的备份
- 比重新处理更省时

### 4. Validation / 验证
- Always verify dataset after caching
- Check sample quality
- 缓存后始终验证数据集
- 检查样本质量

## Troubleshooting / 故障排除

### Dataset Not Found / 数据集未找到

```
FileNotFoundError: Dataset directory not found
```

**Solution / 解决方案:**
- Check the path in config file
- Verify dataset was prepared successfully
- 检查配置文件中的路径
- 验证数据集准备成功

### Incompatible Tokenizer / Tokenizer 不兼容

```
Error loading tokenizer
```

**Solution / 解决方案:**
- Ensure tokenizer_config directory exists in cached dataset
- Re-prepare dataset if needed
- 确保缓存数据集中存在 tokenizer_config 目录
- 如需要重新准备数据集

### Corrupted Dataset / 数据集损坏

```
Error loading arrow files
```

**Solution / 解决方案:**
- Re-prepare the dataset
- Check disk space during preparation
- 重新准备数据集
- 检查准备过程中的磁盘空间

## Summary / 总结

✅ **Your implementation fully supports dataset caching!**

The workflow:
1. **Prepare once**: Convert → Tokenize → Cache as HF Dataset
2. **Use anywhere**: Load cached dataset directly for training
3. **Transfer easily**: Move cached dataset between machines

✅ **你的实现完全支持数据集缓存！**

工作流程：
1. **准备一次**：转换 → 分词 → 缓存为 HF Dataset
2. **任意使用**：训练时直接加载缓存数据集
3. **轻松传输**：在机器之间移动缓存数据集

---

**Configured paths / 已配置路径:**
- Dataset cache: `/Users/lutaojiang/Desktop/project/InfiniGen-NPR/data/infinigen_yaml_226k`
- Ready to use immediately / 立即可用

