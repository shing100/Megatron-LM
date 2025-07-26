# Qwen3 14B MoE Upcycling Example

This example demonstrates how to convert a dense Hugging Face checkpoint of Qwen3-14B into a Mixture-of-Experts (MoE) model using Megatron-Core's upcycling utility, and then start finetuning.

1. Convert the Hugging Face checkpoint to Megatron format:

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

2. Launch training with MoE upcycling enabled:

```bash
bash examples/qwen/train_qwen3_14b_moe_upcycling.sh /path/to/moe-checkpoint ${MEGATRON_FORMAT_DIR} ${TOKENIZER_MODEL} /path/to/data
```

The script loads the dense checkpoint from `--load`, converts it to the MoE format on the fly and stores the converted model under `--save` before training resumes.
