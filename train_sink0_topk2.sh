#!/bin/bash

# ============================================================================
# Experiment B: sink_size=0, memory_top_k=2
# No fixed sink block, dynamically retrieve 2 most similar blocks from memory bank
# ============================================================================

# Project path and config
CONFIG=configs/train_sink0_topk2.yaml
LOGDIR=logs_sink0_topk2
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG
echo "[Exp B] sink_size=0, memory_top_k=2"
echo "Using LoRA mode (training directly from Stage 1)"

# ============ Volcano Engine MLP platform auto-configuration ============

# Number of nodes
NNODES=${MLP_WORKER_NUM:-6}

# GPUs per node
NPROC_PER_NODE=${MLP_WORKER_GPU:-8}

# Master node IP (worker_0)
MASTER_ADDR=${MLP_WORKER_0_HOST:-"192.168.23.159"}

# Communication port (fixed to avoid conflicts with MLP port 2222)
MASTER_PORT=${MASTER_PORT:-29500}

# Automatically get NODE_RANK for the current node
# Determine it by comparing the current node IP with MLP_WORKER_*_HOST
if [ -n "$MLP_ROLE_INDEX" ]; then
    NODE_RANK=$MLP_ROLE_INDEX
elif [ -n "$MLP_WORKER_ALL_HOSTS" ]; then
    # Get the current node IP
    CURRENT_IP=$(hostname -I | awk '{print $1}')
    
    # Find the current node position in the worker list
    NODE_RANK=0
    IFS=',' read -ra HOSTS <<< "$MLP_WORKER_ALL_HOSTS"
    for i in "${!HOSTS[@]}"; do
        if [ "${HOSTS[$i]}" == "$CURRENT_IP" ]; then
            NODE_RANK=$i
            break
        fi
    done
    echo "自动检测: 当前节点 IP=$CURRENT_IP, NODE_RANK=$NODE_RANK"
else
    NODE_RANK=${NODE_RANK:-0}
fi

# ============ Print configuration information ============
echo "========== 火山引擎 MLP 分布式训练配置 =========="
echo "NNODES=$NNODES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "MLP_WORKER_ALL_HOSTS=$MLP_WORKER_ALL_HOSTS"
echo ""
echo "========== [Exp B] sink_size=0, memory_top_k=2 =========="
echo "Training LoRA directly on Stage 1 checkpoint"
echo "========================================================="

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --disable-wandb
