#!/bin/bash
# LoRA训练启动脚本 - Qwen-8B 16K上下文
# 使用此脚本进行LoRA微调，显存需求较小

set -e  # 遇到错误立即退出

CONFIG="configs/train.yaml"

echo "=========================================="
echo "LoRA微调 - Qwen-8B"
echo "=========================================="


# 设置环境变量以减少日志输出
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false


# 启动训练
accelerate launch \
    --config_file configs/accelerate_config_deepspeed_stage3.yaml \
    train.py \
    --config "$CONFIG" || true


python ../gpu_test.py > /dev/null 2>&1