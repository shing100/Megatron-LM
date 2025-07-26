#!/bin/bash

# Runs Qwen3 1.6B model with MoE upcycling

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$1   # path to converted checkpoint
TOKENIZER_MODEL=$2   # path to tokenizer.model

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 5492
    --num-attention-heads 16
    --seq-length 4096
    --max-position-embeddings 4096
    --position-embedding-type rope
    --normalization rmsnorm
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
)

MOE_ARGS=(
    --num-experts 4
    --expert-model-parallel-size 1
    --moe-use-upcycling
    --moe-upcycling-granularity 1
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --mock-data
    --split 99,1,0
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters 100
    --lr 1e-4
    --min-lr 1e-5
    --lr-decay-style cosine
    --bf16
)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval 50
    --eval-interval 50
    --eval-iters 10
    --save ${CHECKPOINT_PATH}
    --load ${CHECKPOINT_PATH}
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}


