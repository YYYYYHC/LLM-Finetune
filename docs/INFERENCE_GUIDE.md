# Inference 使用指南

本文档详细说明如何使用微调后的模型进行推理，生成JSON格式的3D场景文件。

## 📋 目录

- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [参数详解](#参数详解)
- [输出文件说明](#输出文件说明)
- [常见问题](#常见问题)
- [最佳实践](#最佳实践)

## 🚀 快速开始

### 方式1: 使用Shell脚本（推荐）

最简单的方式是使用提供的shell脚本：

```bash
# 使用LoRA模型生成100个场景
bash scripts/generate_json.sh lora 100

# 使用全量微调模型生成50个场景
bash scripts/generate_json.sh full 50
```

### 方式2: 直接使用Python脚本

更灵活的方式，可以自定义所有参数：

```bash
# LoRA模型
python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --num_samples 100 \
    --output_dir ./generated_scenes

# 全量微调模型
python examples/inference_generate_json.py \
    --model_path ./outputs/full_finetune/final \
    --num_samples 100 \
    --output_dir ./generated_scenes
```

## 📖 详细使用说明

### 基本命令结构

```bash
python examples/inference_generate_json.py \
    --model_path <模型路径> \
    [--use_lora] \
    [--num_samples <数量>] \
    [--output_dir <输出目录>] \
    [其他参数...]
```

### 必需参数

- `--model_path`: 微调后的模型路径
  - LoRA模型: `./outputs/lora_finetune/final`
  - 全量微调模型: `./outputs/full_finetune/final`

### 常用可选参数

#### 模型参数

- `--use_lora`: 是否使用LoRA模型（添加此标志表示是）
- `--base_model`: 基础模型名称（仅LoRA需要，默认：`Qwen/Qwen2.5-8B`）

#### 生成参数

- `--num_samples`: 生成样本数量（默认：10）
- `--start_seed`: 起始随机种子（默认：42）
- `--max_new_tokens`: 最大生成token数（默认：8192）
- `--temperature`: 采样温度（默认：0.8，范围：0.1-2.0）
- `--top_p`: Nucleus采样阈值（默认：0.95）
- `--top_k`: Top-k采样（默认：50）
- `--repetition_penalty`: 重复惩罚（默认：1.1）

#### 输出参数

- `--output_dir`: 输出目录（默认：`./generated_scenes`）
- `--save_failed`: 保存失败的生成结果（默认启用）
- `--compact_json`: 使用紧凑格式（不缩进，节省空间）

## 🎛️ 参数详解

### Temperature（温度）

控制生成的随机性和多样性：

- **0.1 - 0.5**: 更确定性，生成更一致的结果
- **0.6 - 0.8**: 平衡的创造性（推荐用于生产）
- **0.9 - 1.2**: 更高的多样性和创造性
- **> 1.2**: 非常随机，可能产生不稳定的输出

**示例**：
```bash
# 保守生成（更一致）
python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --temperature 0.5

# 创意生成（更多样）
python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --temperature 1.0
```

### Top-p（Nucleus Sampling）

控制采样的概率质量：

- **0.9**: 保守，只从高概率token中选择
- **0.95**: 平衡（推荐）
- **0.99**: 更宽松，允许更多样的选择

### Top-k

限制每步采样的候选token数量：

- **10-30**: 更保守
- **40-60**: 平衡（推荐）
- **> 100**: 更自由

### Repetition Penalty

防止重复生成相同的内容：

- **1.0**: 无惩罚
- **1.1**: 轻微惩罚（推荐）
- **1.2-1.5**: 中等惩罚
- **> 1.5**: 强烈惩罚（可能影响质量）

## 📁 输出文件说明

### 目录结构

```
generated_scenes/
├── scene_000000_seed_42.json          # 成功生成的场景
├── scene_000001_seed_43.json
├── ...
├── generation_stats.json              # 统计信息
├── generation_config.json             # 生成配置
└── failed/                            # 失败的生成（如果有）
    ├── scene_seed_100_invalid_json.txt
    └── scene_seed_101_invalid_structure.json
```

### 场景JSON文件

每个成功生成的场景都保存为单独的JSON文件，格式与训练数据一致。

**文件命名规则**：
- `scene_<序号>_seed_<种子>.json`
- 序号：6位数字，从000000开始
- 种子：生成时使用的随机种子

**内容格式**（根据你的训练数据）：
```json
{
  "scene_id": "...",
  "objects": [...],
  "lighting": {...},
  "camera": {...},
  ...
}
```

### 统计信息文件

`generation_stats.json` 包含详细的生成统计：

```json
{
  "total": 100,
  "successful": 95,
  "failed": 5,
  "invalid_json": 2,
  "invalid_structure": 3,
  "avg_chars": 15234,
  "avg_tokens": 3456,
  "avg_generation_time": 2.34,
  "successful_seeds": [42, 43, ...],
  "failed_seeds": [100, 101, ...]
}
```

### 配置文件

`generation_config.json` 记录了生成时使用的所有参数，便于复现。

## 🔧 使用示例

### 示例1: 小规模测试

快速生成少量场景进行测试：

```bash
python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --num_samples 5 \
    --output_dir ./test_scenes
```

### 示例2: 大规模生成

生成大量场景用于生产：

```bash
python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --num_samples 1000 \
    --output_dir ./production_scenes \
    --compact_json \
    --temperature 0.8
```

### 示例3: 高多样性生成

增加创造性和多样性：

```bash
python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --num_samples 100 \
    --temperature 1.0 \
    --top_p 0.98 \
    --top_k 80 \
    --output_dir ./diverse_scenes
```

### 示例4: 保守生成

生成更一致的场景：

```bash
python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --num_samples 100 \
    --temperature 0.6 \
    --top_p 0.9 \
    --top_k 30 \
    --output_dir ./consistent_scenes
```

### 示例5: 指定种子范围

从特定种子开始生成：

```bash
python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --num_samples 100 \
    --start_seed 1000 \
    --output_dir ./custom_seed_scenes
```

## 🐛 常见问题

### Q1: 生成的JSON格式不正确

**原因**：
- Temperature过高
- max_new_tokens不足
- 模型训练不充分

**解决方案**：
```bash
# 降低temperature
--temperature 0.7

# 增加max_new_tokens
--max_new_tokens 10240

# 检查训练loss是否收敛
```

### Q2: 生成速度太慢

**原因**：
- max_new_tokens设置过大
- GPU利用率不足

**解决方案**：
```bash
# 减少max_new_tokens（如果场景不需要那么长）
--max_new_tokens 4096

# 确保模型加载到GPU
# 检查: nvidia-smi
```

### Q3: 生成的场景缺乏多样性

**原因**：
- Temperature过低
- 使用相同的种子范围

**解决方案**：
```bash
# 提高temperature
--temperature 0.9

# 使用不同的start_seed
--start_seed 10000

# 增加top_k和top_p
--top_k 80 --top_p 0.98
```

### Q4: 内存不足（OOM）

**原因**：
- max_new_tokens过大
- 批量生成时GPU内存累积

**解决方案**：
```bash
# 减少max_new_tokens
--max_new_tokens 4096

# 分批生成
bash scripts/generate_json.sh lora 50  # 先生成50个
bash scripts/generate_json.sh lora 50  # 再生成50个
```

### Q5: 生成包含重复内容

**原因**：
- repetition_penalty过低

**解决方案**：
```bash
# 增加repetition_penalty
--repetition_penalty 1.2
```

## 💡 最佳实践

### 1. 开发阶段

在开发和测试时：

```bash
python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --num_samples 10 \
    --temperature 0.8 \
    --output_dir ./dev_test
```

### 2. 生产阶段

在生产环境中：

```bash
python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --num_samples 1000 \
    --temperature 0.75 \
    --compact_json \
    --output_dir ./production_$(date +%Y%m%d)
```

### 3. 参数调优流程

1. **先小规模测试**（10个样本）
2. **调整temperature**找到最佳平衡
3. **确认质量**后再大规模生成
4. **记录成功的参数组合**

### 4. 质量检查

```bash
# 生成后检查统计
cat ./generated_scenes/generation_stats.json

# 查看失败案例
ls ./generated_scenes/failed/

# 随机抽查几个文件
python -m json.tool ./generated_scenes/scene_000000_seed_42.json
```

### 5. 批量处理

对于大量生成需求，建议分批进行：

```bash
# 批次1
bash scripts/generate_json.sh lora 500

# 检查结果
# 如果质量满意，继续

# 批次2
bash scripts/generate_json.sh lora 500

# ...
```

## 🔍 验证生成质量

### 自动验证

脚本会自动进行基本验证：
- JSON格式正确性
- 基本结构完整性

### 手动验证

```bash
# 查看生成的文件
ls generated_scenes/

# 检查具体内容
cat generated_scenes/scene_000000_seed_42.json

# 格式化查看（如果使用了compact_json）
python -m json.tool generated_scenes/scene_000000_seed_42.json

# 统计信息
cat generated_scenes/generation_stats.json
```

### 质量指标

- **成功率**: successful / total
  - 目标：> 95%
  - 如果 < 90%，需要调整参数

- **平均token数**: avg_tokens
  - 应该与训练数据相近
  - 如果差异很大，检查max_new_tokens设置

- **生成时间**: avg_generation_time
  - 正常范围：1-5秒/样本（取决于长度）

## 📊 性能优化

### GPU利用率优化

```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 如果GPU利用率不足，可能是IO瓶颈
# 使用更快的存储或减少保存频率
```

### 并行生成

如果有多个GPU，可以并行运行多个进程：

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --num_samples 500 \
    --start_seed 0 \
    --output_dir ./scenes_gpu0 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python examples/inference_generate_json.py \
    --model_path ./outputs/lora_finetune/final \
    --use_lora \
    --num_samples 500 \
    --start_seed 500 \
    --output_dir ./scenes_gpu1 &

wait
```

## 🎯 总结

- **快速开始**: 使用 `bash scripts/generate_json.sh`
- **灵活控制**: 使用 `python examples/inference_generate_json.py`
- **调整参数**: 根据需求调整temperature、top_p等
- **检查质量**: 查看统计信息和失败案例
- **大规模生成**: 分批处理，并行运行

## 📞 需要帮助？

如果遇到问题：
1. 检查本文档的"常见问题"部分
2. 查看失败案例：`./generated_scenes/failed/`
3. 检查统计信息：`./generated_scenes/generation_stats.json`
4. 尝试调整参数（特别是temperature）

---

**更新时间**: 2025年11月
**版本**: 1.0

