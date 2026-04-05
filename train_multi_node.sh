#!/bin/bash
# 全量微调启动脚本 - Qwen-8B 16K上下文
# 使用此脚本进行全量微调，需要大显存或FSDP/DeepSpeed
# 用法: ./task_16H20.sh <machine_rank>
#   机器0（主节点）: ./task_16H20.sh 0
#   机器1（从节点）: ./task_16H20.sh 1

set -e  # 遇到错误立即退出

# ===== 多机训练配置 =====
MASTER_ADDR="30.217.99.138"  # 主节点IP，请根据实际情况修改
MASTER_PORT=29500
NUM_MACHINES=2
NUM_PROCESSES=16              # 总GPU数：2机×8卡=16

# 从命令行参数获取 machine_rank
MACHINE_RANK=$1

if [ -z "$MACHINE_RANK" ]; then
    echo "错误: 请指定 machine_rank"
    echo "用法: $0 <machine_rank>"
    echo "  主节点: $0 0"
    echo "  从节点: $0 1"
    exit 1
fi

CONFIG="configs/train.yaml"

echo "==========================================="
echo "两机分布式训练 - Qwen-32B"
echo "Machine Rank: $MACHINE_RANK"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "==========================================="


# 设置环境变量以减少日志输出
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

export HF_HUB_CACHE='/root/models'

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export TP_SOCKET_IFNAME=bond1
export NCCL_IB_GDR_LEVEL=0


NCCL_IB_GDR_LEVEL=0 NCCL_NET_GDR_LEVEL=0 \
accelerate launch \
    --config_file configs/accelerate_config_deepspeed_stage3_multi-nodes.yaml \
    --num_machines $NUM_MACHINES \
    --num_processes $NUM_PROCESSES \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    train.py \
    --config "$CONFIG" || true


python ../gpu_test.py > /dev/null 2>&1