#!/usr/bin/env bash
# ============================================================
# finetune_rag_mps_end2end.sh
# RAG end-to-end fine-tuning for Apple Silicon (M-series Macs)
#
# Hardware target: MacBook Pro M4 Max, 48 GB unified memory
# Accelerator:     MPS (Metal Performance Shaders) for training
# Re-encoding:     CPU  (MPS is busy with the training loop;
#                  the M-series CPU handles re-encoding well)
#
# Key differences vs. the original NVIDIA script:
#   --accelerator mps --devices 1  (instead of --gpus 2)
#   no --fp16                      (MPS doesn't support APEX fp16)
#   --precision 32                 (stable; change to bf16-mixed if desired)
#   --train_batch_size 2           (single device; use grad accumulation)
#   --num_retrieval_workers 1      (one Ray actor is enough for single device)
#   --gpu_order "[]"               (not used; CPU handles re-encoding)
# ============================================================

set -euo pipefail

export PYTHONPATH="../":"${PYTHONPATH}"

# Fix: PyTorch and FAISS both ship libomp.dylib on macOS.
# Without this flag the second initialisation aborts the process.
export KMP_DUPLICATE_LIB_OK=TRUE
# Disable Rust tokenizer parallelism — it conflicts with macOS spawn-based
# multiprocessing and causes segfaults during KB re-encoding.
export TOKENIZERS_PARALLELISM=false

# ─── CONFIGURE THESE PATHS ───────────────────────────────────────────────────
# Replace these placeholders with your actual directories before running.

DATA_DIR="/path/to/your/training-data"      # dir with train.source / train.target / val.* / test.*
OUTPUT_DIR="/path/to/your/model-checkpoints"
KB_CSV="/path/to/your/knowledge-base.csv"   # TSV with columns: title, text
KB_DIR="/path/to/your/kb-output"            # where the FAISS index + HF dataset will be saved
SHARD_DIR="./test_run/kb-shards"            # temp dir for re-encoding shards (auto-cleaned at start)
# ─────────────────────────────────────────────────────────────────────────────

echo "==> Step 1: Building custom knowledge base (FAISS index + HF dataset)"
python use_own_knowledge_dataset.py \
    --csv_path "$KB_CSV" \
    --output_dir "$KB_DIR"

echo "==> Step 2: Starting single-node Ray cluster (retriever actors)"
ray start --head

echo "==> Step 3: Running end-to-end RAG fine-tuning on Apple Silicon MPS"
python finetune_rag.py \
    --data_dir              "$DATA_DIR" \
    --output_dir            "$OUTPUT_DIR" \
    --model_name_or_path    facebook/rag-token-base \
    --model_type            rag_token \
    --accelerator           mps \
    --devices               1 \
    --precision             32 \
    --do_train \
    --end2end \
    --do_predict \
    --n_val                 -1 \
    --train_batch_size      2 \
    --eval_batch_size       1 \
    --max_source_length     128 \
    --max_target_length     25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing       0.1 \
    --dropout               0.1 \
    --attention_dropout     0.1 \
    --weight_decay          0.001 \
    --adam_epsilon          1e-08 \
    --max_grad_norm         0.1 \
    --lr_scheduler          polynomial \
    --learning_rate         3e-05 \
    --num_train_epochs      10 \
    --warmup_steps          500 \
    --gradient_accumulation_steps 8 \
    --distributed_retriever ray \
    --num_retrieval_workers 1 \
    --passages_path         "$KB_DIR/my_knowledge_dataset" \
    --index_path            "$KB_DIR/my_knowledge_dataset_hnsw_index.faiss" \
    --index_name            custom \
    --context_encoder_name  facebook/dpr-ctx_encoder-multiset-base \
    --csv_path              "$KB_CSV" \
    --index_gpus            1 \
    --gpu_order             "[]" \
    --shard_dir             "$SHARD_DIR" \
    --indexing_freq         500

echo "==> Step 4: Stopping Ray cluster"
ray stop

echo "==> Done. Checkpoints saved to $OUTPUT_DIR"

# ─── NOTES ───────────────────────────────────────────────────────────────────
#
# MEMORY:
#   RAG-token-base (BART-large + DPR) requires ~6–8 GB for float32.
#   With 48 GB unified memory, train_batch_size can be increased to 4–8.
#
# PRECISION:
#   --precision 32  → safest, ~8 GB model footprint
#   --precision bf16-mixed → ~half the memory, M4 Max supports BF16 natively
#   DO NOT use --fp16 (APEX NVIDIA-only; not supported on MPS)
#
# RE-ENCODING SPEED:
#   CPU re-encoding is slower than a dedicated GPU.
#   For initial experiments, use a smaller KB subset:
#     10 000–50 000 passages instead of 250 000
#   Reduce indexing_freq if re-encoding finishes before the next trigger:
#     --indexing_freq 200
#
# DEBUGGING:
#   To run a quick sanity check before a full training run:
#     add --fast_dev_run to the python call above
#   To log to Weights & Biases:
#     add --logger_name wandb
#
# ─────────────────────────────────────────────────────────────────────────────
