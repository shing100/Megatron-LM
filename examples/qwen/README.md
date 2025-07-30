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

## 2. Preprocess a dataset

If you have a JSONL dataset with a `text` column, convert it to Megatron's
binary format using `tools/preprocess_data.py`:

```bash
python tools/preprocess_data.py \
    --input /path/to/my_corpus.jsonl \
    --json-keys text \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --output-prefix /path/to/my_dataset_text_document \
    --append-eod
```

The command produces `.bin` and `.idx` files prefixed with
`my_dataset_text_document` that can be passed to `--data-path` during training.

## 3. Train with MoE upcycling

Use the provided shell script to launch training from the repository root.
It loads the converted checkpoint and enables MoE upcycling at runtime.

```bash
bash examples/qwen/train_qwen3_1_6b_moe_upcycling.sh \
    /path/to/qwen3-1_6b-mcore \
    /path/to/tokenizer.model \
    /path/to/my_dataset_text_document
```

The script sets up distributed arguments for an 8–GPU node and trains the
model with `--moe-use-upcycling`.
