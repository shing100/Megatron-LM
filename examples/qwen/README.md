# Qwen3 1.6B MoE Upcycling Example

This example shows how to convert the **Qwen3 1.6B** checkpoint from
HuggingFace format to Megatron format and train it using the MoE
upcycling feature. The script assumes a single node with eight GPUs.

## 1. Convert the HuggingFace checkpoint

```bash
HF_FORMAT_DIR=/path/to/qwen3-1_6b-hf
MEGATRON_FORMAT_DIR=/path/to/qwen3-1_6b-mcore
TOKENIZER_MODEL=/path/to/tokenizer.model

python tools/checkpoint/convert.py \
    --bf16 \
    --model-type GPT \
    --loader llama_mistral \
    --saver core \
    --target-tensor-parallel-size 1 \
    --checkpoint-type hf \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --model-size llama3
```

## 2. Train with MoE upcycling

Use the provided shell script to launch training. It loads the converted
checkpoint and enables MoE upcycling at runtime.

```bash
bash examples/qwen/train_qwen3_1_6b_moe_upcycling.sh \
    /path/to/qwen3-1_6b-mcore \
    /path/to/tokenizer.model
```

The script sets up distributed arguments for an 8–GPU node and trains the
model with `--moe-use-upcycling`.
