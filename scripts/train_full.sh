#!/bin/bash
# 全量微调启动脚本 - Qwen-8B 16K上下文
# 使用此脚本进行全量微调，需要大显存或FSDP/DeepSpeed

set -e  # 遇到错误立即退出

CONFIG="configs/full_finetune_qwen8b.yaml"

echo "=========================================="
echo "全量微调 - Qwen-8B"
echo "=========================================="


# 设置环境变量以减少日志输出
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

# 启动训练 - 使用FSDP配置
accelerate launch \
    --config_file configs/accelerate_config_fsdp.yaml \
    train.py \
    --config "$CONFIG"


