# RAG End-to-End Retriever — Apple Silicon Adaptation

> **Note:** This repository is a fork/adaptation of the original work by Shamane Siriwardhana et al.
> The original README from the authors is preserved at [README_original_authors.md](./README_original_authors.md).
> The original paper: *Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering*, TACL 2023.

---

## About This Fork

The modifications in this repository are made by **Luis Manuel Manrique** ([@lmmanriquem](https://github.com/lmmanriquem)) for personal research and experimentation.

The original codebase targets NVIDIA CUDA hardware exclusively. The goal of this adaptation is to make the full end-to-end RAG training pipeline run on **Apple Silicon (M-series)** hardware, while fully preserving the original NVIDIA/CUDA code path so the same repository works on both platforms without modification.

---

## Experiment Hardware

| Component | Specification |
|---|---|
| Machine | MacBook Pro M4 Max |
| Unified Memory | 48 GB |
| GPU Cores | 40 (Apple Silicon GPU) |
| CPU Cores | 16 (12 performance + 4 efficiency) |
| Architecture | arm64 (Apple Silicon) |
| OS | macOS Sequoia 15.x |
| GPU Backend | MPS (Metal Performance Shaders) via PyTorch |
| Re-encoding | CPU (MPS is occupied by the training loop) |

---

## Changes Made to the Original Codebase

### Modified files

**`finetune_rag.py`**
- Removed top-level `pynvml` import; replaced with lazy runtime import inside CUDA branch only
- `training_step`: GPU detection is now platform-aware — CUDA branch preserved intact, new MPS/CPU branch added for Apple Silicon
- State dict transfer: explicit `.cpu()` before `load_state_dict()` in re-encoding child processes (required for MPS)
- Retriever access path corrected: `self.model.rag.retriever.re_load()` / `self.model.rag.retriever.init_retrieval()`
- Three `hparams.gpus` None guards to prevent crashes when `--gpus` is not set (MPS path)

**`lightning_base.py`**
- `AdamW` import moved from `transformers` to `torch.optim` (removed from transformers in 4.x)
- `total_steps()`: None-safe `gpus` guard
- `generic_train()`: fp16 blocked on MPS with a clear warning; bf16-mixed still allowed
- `ModelCheckpoint`: migrated from deprecated `filepath=` to `dirpath=` + `filename=`
- Multi-GPU DDP guard: strategy only set when `gpus > 1`

**`kb_encode_utils.py`**
- FAISS thread count is now platform-aware: capped at 8 on Apple Silicon, uses all cores on Linux/NVIDIA

**`requirements.txt`**
- Added version upper bounds: `transformers < 5.0.0`, `datasets < 3.0.0`
- `nvidia-ml-py3` marked as optional/NVIDIA-only with install instructions
- `setup_env.py` recommended as the install entry point

### New files

**`finetune_rag_mps_end2end.sh`** — Launch script for Apple Silicon, mirrors the original NVIDIA script with MPS-specific flags (`--accelerator mps --devices 1 --precision 32`, no `--fp16`)

**`setup_env.py`** — Platform-detecting installer. Auto-detects Apple Silicon vs NVIDIA vs CPU-only and installs the correct dependencies. Supports `--dry-run` flag.

---

## Step-by-Step Setup Guide (Apple Silicon — macOS)

This guide documents the exact steps followed to get the pipeline running on a MacBook Pro M4 Max. Follow these in order.

### Prerequisites

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Internet connection (model downloads ~2.5 GB total on first run)
- VS Code or any terminal

---

### Step 1 — Clone or open the repository

Open a terminal and navigate to the repository folder:

```bash
cd /rag-end2end-retriever
```

---

### Step 2 — Create a Python 3.11 conda environment

> **Important:** Python 3.13+ is NOT compatible with the dependency stack (PyTorch 2.x). Use Python 3.11.

```bash
conda create -n rag-env python=3.11
conda activate rag-env
python --version   # must show Python 3.11.x
```

Expected prompt: `(rag-env) your-machine %`

---

### Step 3 — Install dependencies

Run the platform-detecting installer (do NOT use `pip install -r requirements.txt` directly):

```bash
python setup_env.py
```

The script will:
- Detect Apple Silicon automatically
- Install PyTorch with MPS support
- Skip `nvidia-ml-py3` (NVIDIA-only)
- Verify that MPS is available

Expected output at the end:
```
✓ MPS  available  — Apple Silicon GPU will be used for training
── Setup complete ───────────────────────────────────────────
```

---

### Step 4 — Pin dependency versions (important)

`pip` may resolve newer incompatible versions. Force the correct range:

```bash
pip install "transformers>=4.30.0,<5.0.0" "datasets>=2.10.0,<3.0.0"
```

---

### Step 5 — Verify installation

```bash
python -c "
import torch, pytorch_lightning as pl, transformers, datasets, faiss, ray
print(f'torch:             {torch.__version__}')
print(f'MPS available:     {torch.backends.mps.is_available()}')
print(f'pytorch-lightning: {pl.__version__}')
print(f'transformers:      {transformers.__version__}')
print(f'datasets:          {datasets.__version__}')
from transformers import RagSequenceForGeneration, RagTokenForGeneration, RagRetriever
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
print('RAG classes:       OK')
"
```

All lines should print without errors. `MPS available` must be `True`.

---

### Step 6 — Prepare training data

Training data must be in plain text files, one item per line:

```
data/
  train.source   ← one question per line
  train.target   ← one answer per line  (same order as .source)
  val.source
  val.target
  test.source
  test.target
```

Example (smoke test with dummy data):

```bash
mkdir -p smoke_test/data smoke_test/kb smoke_test/output smoke_test/shards

python - << 'EOF'
import os

questions = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water?",
    "How many planets are in the solar system?",
    "What language do Brazilians speak?",
]
answers = ["Paris", "Shakespeare", "100 degrees Celsius", "eight", "Portuguese"]

for split, q, a in [("train", questions, answers),
                    ("val",   questions[:2], answers[:2]),
                    ("test",  questions[:2], answers[:2])]:
    open(f"smoke_test/data/{split}.source","w").write("\n".join(q)+"\n")
    open(f"smoke_test/data/{split}.target","w").write("\n".join(a)+"\n")

passages = [
    ("France",       "Paris is the capital and largest city of France."),
    ("Shakespeare",  "William Shakespeare wrote Romeo and Juliet around 1594."),
    ("Water",        "Water boils at 100 degrees Celsius at sea level."),
    ("Solar System", "The solar system has eight planets."),
    ("Brazil",       "Portuguese is the official language of Brazil."),
    ("Science",      "Physics studies matter energy and the fundamental forces of nature."),
    ("History",      "The Roman Empire fell in 476 AD."),
    ("Mathematics",  "The square root of 144 is 12."),
]
with open("smoke_test/kb/passages.tsv","w") as f:
    for title, text in passages:
        f.write(f"{title}\t{text}\n")

print("(GOOD) Data files created . OK")
EOF
```

---

### Step 7 — Encode the knowledge base

The knowledge base must be encoded with DPR before training. On Apple Silicon, use this script (bypasses the `dataset.map()` multiprocessing issue that causes segfaults on macOS):

```bash
python - << 'EOF'
import os, torch, faiss, numpy as np
from datasets import Dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Encoding on: {device}")

titles, texts = [], []
with open("smoke_test/kb/passages.tsv") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            titles.append(parts[0]); texts.append(parts[1])

ctx_encoder  = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base").to(device)
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
ctx_encoder.eval()

all_emb = []
with torch.no_grad():
    for i in range(0, len(titles), 4):
        inp = ctx_tokenizer(titles[i:i+4], texts[i:i+4],
                            truncation=True, padding="longest",
                            max_length=512, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        all_emb.append(ctx_encoder(**inp, return_dict=True).pooler_output.cpu().float().numpy())

embeddings = np.concatenate(all_emb)
ds = Dataset.from_dict({"title": titles, "text": texts,
                        "embeddings": [e for e in embeddings]})
ds.save_to_disk("smoke_test/kb/my_knowledge_dataset")
print("(GOOD)) Dataset with embeddings saved . OK")
EOF
```

Then build the FAISS index in a **separate step** (avoids the dual OpenMP conflict between PyTorch and FAISS):

```bash
KMP_DUPLICATE_LIB_OK=TRUE python - << 'EOF'
import faiss
from datasets import load_from_disk

faiss.omp_set_num_threads(1)
ds = load_from_disk("smoke_test/kb/my_knowledge_dataset")
index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
ds.add_faiss_index("embeddings", custom_index=index)
ds.get_index("embeddings").save("smoke_test/kb/my_knowledge_dataset_hnsw_index.faiss")
print("(GOOD)) FAISS index saved - OK")
EOF
```

> **Why two steps?** PyTorch (via MPS) and FAISS both ship their own `libomp.dylib` on macOS. Loading both in the same process causes an abort. Splitting the steps keeps them in separate processes.

---

### Step 8 — Start the Ray cluster

```bash
ray start --head
```

Expected output:
```
Ray runtime started.
```

---

### Step 9 — Run training

For a quick smoke test (1 train batch + 1 val batch, no checkpoint saved):

```bash
KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false python finetune_rag.py \
    --data_dir              smoke_test/data \
    --output_dir            smoke_test/output \
    --model_name_or_path    facebook/rag-token-base \
    --model_type            rag_token \
    --accelerator           mps \
    --devices               1 \
    --precision             32 \
    --do_train \
    --end2end \
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
    --num_train_epochs      1 \
    --warmup_steps          0 \
    --gradient_accumulation_steps 1 \
    --distributed_retriever ray \
    --num_retrieval_workers 1 \
    --passages_path         smoke_test/kb/my_knowledge_dataset \
    --index_path            smoke_test/kb/my_knowledge_dataset_hnsw_index.faiss \
    --index_name            custom \
    --context_encoder_name  facebook/dpr-ctx_encoder-multiset-base \
    --csv_path              smoke_test/kb/passages.tsv \
    --index_gpus            1 \
    --gpu_order             "[]" \
    --shard_dir             smoke_test/shards \
    --indexing_freq         500 \
    --fast_dev_run
```

For full training with your own data, edit and run `finetune_rag_mps_end2end.sh` (update the path variables at the top of the script).

---

### Step 10 — Stop the Ray cluster when done

```bash
ray stop
```

---

## Environment Variables — macOS Required

These must be set before any run that combines PyTorch + FAISS on macOS:

| Variable | Value | Reason |
|---|---|---|
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | PyTorch and FAISS both ship `libomp.dylib`; without this flag, the second load aborts the process |
| `TOKENIZERS_PARALLELISM` | `false` | Rust-based HuggingFace tokenizers conflict with macOS `spawn`-based multiprocessing during KB re-encoding |

Both are already exported in `finetune_rag_mps_end2end.sh`.

---

## Known Issues on macOS ARM64

| Issue | Cause | Fix applied |
|---|---|---|
| `zsh: segmentation fault` during KB encoding | `dataset.map()` + tokenizer parallelism on macOS | Custom encoding loop (Step 7) |
| `OMP: Error #15` / abort during FAISS index build | Dual `libomp.dylib` (PyTorch + FAISS) | Separate FAISS step + `KMP_DUPLICATE_LIB_OK=TRUE` |
| `ImportError: cannot import name 'AdamW' from 'transformers'` | `AdamW` removed from transformers in 4.x | Import from `torch.optim` in `lightning_base.py` |
| `--fp16` crash on MPS | APEX not available on Apple Silicon | fp16 blocked in `lightning_base.py`; use `--precision 32` or `bf16-mixed` |

---

## NVIDIA / CUDA Compatibility

The original CUDA code path is fully preserved. On any machine with an NVIDIA GPU:

- `pynvml` is loaded lazily at runtime only when CUDA is detected
- `nvidia-ml-py3` is installed by `setup_env.py` only on NVIDIA systems
- DDP multi-GPU strategy is activated automatically when `--gpus > 1`
- FAISS uses all available CPU cores for indexing (no cap)

No changes are required to run on NVIDIA hardware. The same codebase handles both.

---

## Reference

- Original README and implementation details: [README_original_authors.md](./README_original_authors.md)
- Original paper: Siriwardhana et al., *Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering*, TACL 2023. [arXiv:2210.02627](https://arxiv.org/abs/2210.02627)
- Original blog post: [How to finetune the entire RAG architecture including DPR retriever](https://shamanesiri.medium.com/how-to-finetune-the-entire-rag-architecture-including-dpr-retriever-4b4385322552)
- Base RAG paper: Lewis et al., [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
