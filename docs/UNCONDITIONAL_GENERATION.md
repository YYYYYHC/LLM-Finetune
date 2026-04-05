# Unconditional Generation Guide / 无条件生成指南

## Overview / 概述

This project is now configured for **unconditional scene generation**: the model learns to generate diverse 3D scene JSON files by changing only the random seed, without requiring specific input conditions.

本项目现已配置为**无条件场景生成**：模型学习生成多样化的3D场景JSON文件，仅通过改变随机种子即可，无需特定输入条件。

## 🎯 Training Objective / 训练目标

**Goal**: Given a simple trigger prompt, generate complete and diverse scene JSON files.

**目标**：给定简单触发词，生成完整且多样化的场景JSON文件。

```python
# Training format / 训练格式
User: "Generate a 3D scene in JSON format:"
Assistant: {完整的场景JSON，1000-2000行}

# Inference / 推理
seed=42  → Scene A
seed=123 → Scene B (completely different)
seed=999 → Scene C (another unique scene)
```

## 📊 Data Format / 数据格式

### Training Samples / 训练样本

Each of your 226k YAML files becomes one training sample:

每个226k YAML文件变成一个训练样本：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Generate a 3D scene in JSON format:"
    },
    {
      "role": "assistant", 
      "content": "{\"scene_id\":\"xxx\",\"objects\":[...],\"lighting\":{...},\"camera\":{...}}"
    }
  ]
}
```

**Key features / 关键特性:**
- ✅ Simple trigger prompt / 简单触发词
- ✅ Complete JSON as output / 完整JSON作为输出
- ✅ Compact format (no indent) to save tokens / 紧凑格式节省token
- ✅ Model learns the full generation task / 模型学习完整生成任务

### Why This Format? / 为什么这个格式？

**❌ Old format (理解任务):**
```python
User: "Please analyze this scene: {完整JSON}"  # 输入包含答案！
Assistant: "I understand... {JSON[:200]}"      # 只输出片段
# 问题：模型学不会生成，只会复述
```

**✅ New format (生成任务):**
```python
User: "Generate a 3D scene in JSON format:"   # 简单触发
Assistant: {完整JSON}                         # 完整输出
# 正确：模型学习从零生成完整JSON
```

## ⚙️ Configuration Changes / 配置变更

### 1. Max Length Increased / 序列长度增加

```yaml
# Before / 之前
max_length: 2048  # Too short for 1000-2000 line JSON

# After / 之后  
max_length: 8192  # Can accommodate longer JSON
```

**Why / 为什么:**
- Your JSON files: 1000-2000 lines / 你的JSON：1000-2000行
- Estimated tokens: 5000-10000 tokens / 预估token数
- 8192 provides buffer / 8192提供缓冲

**Can be further increased / 可进一步增加:**
```bash
# Qwen2.5 supports up to 32k tokens
python -m src.data.prepare_dataset --max_length 16384
```

### 2. JSON Format / JSON格式

```python
# Compact format (saves ~30-40% tokens)
json.dumps(item, ensure_ascii=False, separators=(',', ':'))

# Before (with indent=2):
{
  "scene_id": "xxx",
  "objects": [
    {...}
  ]
}

# After (compact):
{"scene_id":"xxx","objects":[{...}]}
```

## 🚀 Usage / 使用方法

### Step 1: Prepare Dataset / 准备数据集

```bash
# Data will be formatted for unconditional generation
bash scripts/prepare_data.sh
```

This will:
1. Load your 226k JSON files / 加载226k JSON文件
2. Convert each to compact JSON format / 转换为紧凑JSON格式
3. Create training samples with simple trigger / 创建带简单触发词的训练样本
4. Tokenize with max_length=8192 / 使用8192最大长度分词

**Expected time / 预期时间:** 1-2 hours with 8 processes / 8进程1-2小时

### Step 2: Train Model / 训练模型

```bash
# LoRA (recommended for first try)
bash scripts/train_lora.sh

# Or full fine-tuning
bash scripts/train_full.sh
```

**Training will teach the model to:**
- Generate valid JSON syntax / 生成有效JSON语法
- Create diverse scene structures / 创建多样化场景结构
- Produce 1000-2000 line outputs / 产生1000-2000行输出
- Maintain consistency across fields / 保持字段一致性

### Step 3: Inference / 推理

Create a new inference script for unconditional generation:

创建无条件生成的推理脚本：

```python
# examples/inference_unconditional.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Load model
model_path = "./outputs/lora_finetune/final"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare prompt
messages = [
    {
        "role": "user",
        "content": "Generate a 3D scene in JSON format:"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate with different seeds for diversity
for seed in [42, 123, 456, 789, 999]:
    torch.manual_seed(seed)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=8192,      # Allow long generation
        temperature=0.8,           # Higher for diversity
        top_p=0.95,               # Nucleus sampling
        do_sample=True,           # Enable sampling
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    
    # Parse and validate JSON
    try:
        scene_json = json.loads(response)
        print(f"\n=== Scene with seed={seed} ===")
        print(f"Valid JSON: {len(json.dumps(scene_json))} characters")
        
        # Save to file
        with open(f"generated_scene_seed_{seed}.json", "w") as f:
            json.dump(scene_json, f, indent=2, ensure_ascii=False)
        
        print(f"Saved to generated_scene_seed_{seed}.json")
        
    except json.JSONDecodeError as e:
        print(f"Seed {seed}: Invalid JSON - {e}")
        print(f"Response preview: {response[:200]}...")
```

## 🎲 Generation Parameters / 生成参数

### Key Parameters / 关键参数

```python
model.generate(
    max_new_tokens=8192,    # How long to generate
    temperature=0.8,        # Randomness (0.7-1.0 for diversity)
    top_p=0.95,            # Nucleus sampling threshold
    top_k=50,              # Top-k sampling (optional)
    do_sample=True,        # Must be True for stochastic generation
    seed=42                # Change this for different outputs
)
```

### Temperature Guide / Temperature 指南

| Temperature | Effect / 效果 |
|-------------|--------------|
| 0.5-0.7 | Conservative, safer JSON / 保守，更安全的JSON |
| **0.8** | **Balanced diversity (recommended)** / **平衡的多样性（推荐）** |
| 0.9-1.0 | High diversity, may have errors / 高多样性，可能有错误 |
| 1.2+ | Very creative but risky / 非常创意但有风险 |

### Sampling Strategies / 采样策略

**Option 1: Nucleus Sampling (Recommended) / 核采样（推荐）**
```python
temperature=0.8,
top_p=0.95,
do_sample=True
```

**Option 2: Top-K Sampling**
```python
temperature=0.8,
top_k=50,
do_sample=True
```

**Option 3: Combined**
```python
temperature=0.8,
top_p=0.95,
top_k=50,
do_sample=True
```

## 📈 Expected Results / 预期结果

### After Training / 训练后

✅ **Good signs / 好的迹象:**
- Model generates valid JSON syntax / 模型生成有效JSON语法
- Scenes are diverse with different seeds / 不同种子产生多样场景
- Structure matches training data / 结构匹配训练数据
- Field values are reasonable / 字段值合理

⚠️ **Potential issues / 潜在问题:**
- JSON syntax errors (missing brackets, commas) / JSON语法错误
- Repetitive patterns / 重复模式
- Truncated output / 输出截断
- Invalid field values / 无效字段值

### Generation Quality / 生成质量

**Factors affecting quality / 影响质量的因素:**

1. **Model size / 模型大小**
   - 8B: Good for basic generation / 适合基础生成
   - 14B+: Better for complex, long JSON / 更适合复杂长JSON

2. **Training data quality / 训练数据质量**
   - 226k diverse scenes = good / 226k多样场景=好
   - Need consistency in structure / 需要结构一致性

3. **Training duration / 训练时长**
   - 3 epochs: Minimum / 最少
   - 5-10 epochs: Better for generation / 生成任务更好
   - Monitor overfitting / 监控过拟合

4. **Inference parameters / 推理参数**
   - Temperature: Balance creativity and correctness / 平衡创意和正确性
   - Sampling: Use nucleus or top-k / 使用核采样或top-k

## 🛠️ Post-Processing / 后处理

### JSON Validation / JSON验证

```python
def validate_and_fix_json(response: str) -> dict:
    """
    Validate and attempt to fix common JSON errors.
    """
    try:
        # Try direct parsing
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to fix common issues
        fixed = response
        
        # Remove trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # Add missing closing brackets
        open_braces = fixed.count('{') - fixed.count('}')
        if open_braces > 0:
            fixed += '}' * open_braces
        
        open_brackets = fixed.count('[') - fixed.count(']')
        if open_brackets > 0:
            fixed += ']' * open_brackets
        
        try:
            return json.loads(fixed)
        except:
            raise ValueError("Cannot fix JSON")
```

### Field Validation / 字段验证

```python
def validate_scene_structure(scene: dict) -> bool:
    """
    Validate that generated scene has expected structure.
    """
    required_fields = ["scene_id", "objects", "lighting", "camera"]
    
    for field in required_fields:
        if field not in scene:
            return False
    
    # Check object structure
    if not isinstance(scene["objects"], list):
        return False
    
    # Additional validations...
    return True
```

## 💡 Tips for Better Generation / 更好生成的技巧

### 1. Use Constrained Decoding / 使用受约束解码

Consider using libraries like `lm-format-enforcer` or `guidance` to enforce valid JSON structure during generation.

考虑使用库如 `lm-format-enforcer` 或 `guidance` 在生成时强制有效JSON结构。

### 2. Beam Search for Consistency / Beam Search提高一致性

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=8192,
    num_beams=4,           # Use beam search
    do_sample=False,       # Deterministic
    early_stopping=True
)
```

### 3. Multiple Attempts / 多次尝试

```python
for attempt in range(5):
    output = model.generate(...)
    try:
        scene = json.loads(output)
        if validate_scene_structure(scene):
            return scene  # Success!
    except:
        continue  # Try again
```

### 4. Fine-tune on Failures / 在失败样本上微调

Collect failed generations, manually fix them, and continue training to improve quality.

收集失败的生成，手动修复，继续训练以提高质量。

## 🔍 Monitoring Training / 监控训练

### Key Metrics / 关键指标

1. **Loss / 损失**
   - Should decrease steadily / 应稳步下降
   - For generation: expect 1.0-3.0 / 生成任务：期望1.0-3.0

2. **Perplexity / 困惑度**
   - Lower is better / 越低越好
   - Good: < 20 / 好：< 20

3. **Sample quality / 样本质量**
   - Manually check generated samples during training
   - 训练期间手动检查生成样本

## 📚 Related Documentation / 相关文档

- `docs/DATA_FORMAT.md` - Original data format guide / 原始数据格式指南
- `USAGE.md` - General usage guide / 通用使用指南
- `docs/CONFIGURATION.md` - Configuration reference / 配置参考

## ❓ FAQ

**Q: Why is max_length increased to 8192?**
A: Your JSON files are 1000-2000 lines, which require 5000-10000 tokens. 8192 provides sufficient space.

**Q: 为什么max_length增加到8192？**
A: 你的JSON文件有1000-2000行，需要5000-10000个token。8192提供足够空间。

---

**Q: Will the model always generate valid JSON?**
A: Not guaranteed. Use post-processing and validation. Consider constrained decoding for better results.

**Q: 模型总是生成有效JSON吗？**
A: 不保证。使用后处理和验证。考虑受约束解码以获得更好结果。

---

**Q: How to increase diversity?**
A: Use higher temperature (0.8-1.0), nucleus sampling (top_p=0.95), and different random seeds.

**Q: 如何增加多样性？**
A: 使用更高temperature（0.8-1.0）、核采样（top_p=0.95）和不同随机种子。

---

**Q: Training takes too long, what to do?**
A: Use LoRA instead of full fine-tuning, reduce max_length if possible, or use fewer epochs.

**Q: 训练太久怎么办？**
A: 使用LoRA而非全量微调，如可能减少max_length，或使用更少epoch。

---

## Summary / 总结

Your project is now configured for **unconditional scene generation**:

你的项目现已配置为**无条件场景生成**：

✅ **Data format**: Simple trigger → Complete JSON / 简单触发→完整JSON
✅ **Max length**: 8192 (supports long JSON) / 8192（支持长JSON）
✅ **Compact JSON**: Saves tokens / 紧凑JSON：节省token
✅ **Multi-processing**: Fast dataset preparation / 多进程：快速准备

**Ready to start / 准备开始:**
```bash
bash scripts/prepare_data.sh  # Prepare dataset
bash scripts/train_lora.sh    # Train model
```

**Expected outcome / 预期结果:**
Change random seed → Get different 3D scenes!
改变随机种子 → 获得不同的3D场景！🎲

