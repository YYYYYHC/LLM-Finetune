> ⚠️ **Qwen3-VL LoRA 补丁**：`pip install -r requirements.txt` 后，务必运行下方命令修正 vLLM 的源码前缀，否则 Qwen3-VL + LoRA 会在 `lora_shrink` 处异常。

```bash
python - <<'PY'
import inspect
from pathlib import Path
import vllm.model_executor.models.qwen3_vl as qwen3_vl

file_path = Path(inspect.getfile(qwen3_vl))
text = file_path.read_text()
patched = text.replace(
    'connector="model.visual.merger"',
    'connector="visual.merger"',
).replace(
    'tower_model="model.visual."',
    'tower_model="visual."',
)

if text == patched:
    print(f"No change needed: {file_path}")
else:
    file_path.write_text(patched)
    print(f"Patched: {file_path}")
PY
```

### 🚀 vLLM 极速推理（新）
> 代码位置： [src/inference/vllm_engine.py](src/inference/vllm_engine.py)

当需要高吞吐或低延迟推理时，可使用基于 vLLM 的新脚本：

- 单实例高并发，自动利用 GPU 显存
- 兼容 LoRA / 全量微调模型
- 自动从回答中提取 JSON，失败样本会落到 `failed/`
- `panorama` 模式会检查图片路径并走多模态推理

#### LoRA 示例

```bash
python -m src.inference.vllm_engine \
    --model_path Qwen/Qwen2.5-8B-Instruct \
    --adapter_path ./outputs/lora_roomgen/final \
    --input_path ./demo_inputs/captions.jsonl \
    --mode caption \
    --batch_size 32 \
    --output_dir ./generated_with_vllm
```

#### 全量模型 + 单条 Prompt

```bash
python -m src.inference.vllm_engine \
    --model_path ./outputs/full_finetune/final \
    --prompt "Generate a 3D scene in JSON format:" \
    --output_dir ./generated_single
```

运行后目录结构：

- `responses.jsonl`：完整回答与解析状态
- `json/*.json`：成功解析的 JSON（`--skip_json_files` 可关闭写入）
- `failed/*.txt`：无法解析的原文
- `generation_stats.json`：统计信息（成功率、耗时、Token 数）

**直接喂图片**：若只想给某张全景图或整个目录跑推理，可使用 `--image_path` / `--image_dir`（仅 panorama 模式）。可与 `--prompt` 组合自定义提示，例如：

```bash
python -m src.inference.vllm_engine \
    --mode panorama \
    --model_path Qwen/Qwen2.5-VL-Instruct \
    --adapter_path ./outputs/panorama_lora/final \
    --image_dir ./panorama_images \
    --prompt "根据该全景图生成JSON场景：" \
    --output_dir ./panorama_vllm_dir
```

更多参数（`--max_tokens` / `--stop` / `--gpu_memory_utilization` 等）可通过 `python -m src.inference.vllm_engine --help` 查看。