#!/bin/bash
set -e

# Runs Qwen3 1.6B model with MoE upcycling. The script assumes it is run
# from the root of the Megatron-LM repository. Pass paths to the converted
# checkpoint, tokenizer model, and preprocessed dataset prefix.

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}

PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

CHECKPOINT_PATH=$1   # path to converted checkpoint
TOKENIZER_MODEL=$2   # path to tokenizer.model
DATA_PATH=$3         # prefix for dataset (_text_document)

# Ensure pretrain_gpt.py exists
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: $PRETRAIN_SCRIPT_PATH not found. Run from repository root." >&2
    exit 1
fi

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
    --data-path ${DATA_PATH}
    --split 949,50,1
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

torchrun ${DISTRIBUTED_ARGS[@]} "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}


