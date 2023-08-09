#!/usr/bin/env bash

CONFIG=$1
WORK_DIR=$2
LOAD_FROM=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

((PORT = $PORT + $RANDOM % 50 + 1))

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=4 \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --work-dir $WORK_DIR \
    --cfg-options load_from=$3 \
    --launcher pytorch ${@:4}