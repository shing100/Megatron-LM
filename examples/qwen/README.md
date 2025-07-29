# Qwen3 MoE Upcycling Examples

This example demonstrates how to convert dense Hugging Face checkpoints of Qwen3 models into Mixture-of-Experts (MoE) models using Megatron-Core's upcycling utility and then start finetuning.
The `train_qwen3_14b_moe_upcycling.sh` script uses 64 experts so the resulting model stays under 140B parameters.

## 1. Convert Hugging Face checkpoints

The following commands convert a Hugging Face format checkpoint to Megatron format.

### Qwen3-14B

```bash
TOKENIZER_MODEL=/path/to/tokenizer.model
HF_FORMAT_DIR=/path/to/hf/Qwen3-14B
MEGATRON_FORMAT_DIR=/path/to/megatron/Qwen3-14B-dense

python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader loader_hf \
    --saver mcore \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL}
```

### Qwen3-1.6B

```bash
TOKENIZER_MODEL=/path/to/tokenizer.model
HF_FORMAT_DIR=/path/to/hf/Qwen3-1.6B
MEGATRON_FORMAT_DIR=/path/to/megatron/Qwen3-1.6B-dense

python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader loader_hf \
    --saver mcore \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL}
```

## 2. Launch training with MoE upcycling enabled

### Qwen3-14B

```bash
bash examples/qwen/train_qwen3_14b_moe_upcycling.sh \ 
    /path/to/moe-checkpoint \ 
    /path/to/megatron/Qwen3-14B-dense \ 
    /path/to/tokenizer.model \ 
    /path/to/data
```

### Qwen3-1.6B

```bash
bash examples/qwen/train_qwen3_1_6b_moe_upcycling.sh \ 
    /path/to/moe-checkpoint \ 
    /path/to/megatron/Qwen3-1.6B-dense \ 
    /path/to/tokenizer.model \ 
    /path/to/data
```

The scripts load the dense checkpoints from `--load`, convert them to the MoE format on the fly and store the converted model under `--save` before training resumes.
