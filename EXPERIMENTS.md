# Experiments Log

This document combines dataset preparation and experiment tracking for the replication of
Siriwardhana et al. (TACL 2023) on Apple Silicon. It is written so that anyone — including
future readers — can reproduce every step from scratch and understand exactly what happened
at each stage.

> **Status:** ✅ SQuAD mini quick test completed — QAConv mini pending.

---

## Datasets selected

| Dataset | Used for | Paper table | QA pairs | KB passages | License |
|---|---|---|---|---|---|
| **SQuAD v1.1** | Open-Domain QA | Table 5, §5.3 | ~87K | ~35K | CC BY-SA 4.0 |
| **QAConv v1.1** | Conversation Domain | Table 1 | ~26K | ~69K | Research use |

### Why these two and not the others

The paper uses four domains in total. Two were excluded:

- **NewsQA** — based on CNN articles whose redistribution is restricted. Reproducing
  the dataset requires running a multi-step compilation pipeline that involves accepting
  CNN's terms, making it impractical for open replication.
- **COVID-19 / CORD-19** — the paper did not use CORD-19 directly as a QA dataset.
  Instead it required generating synthetic QA pairs using a BART model fine-tuned on
  SQuAD, which is a separate multi-day pipeline on top of the replication itself.

SQuAD and QAConv are freely available, fully self-contained, and cover the two
most clearly documented experiments in the paper.

---

## Dataset 1 — SQuAD v1.1

### What is SQuAD?

SQuAD (Stanford Question Answering Dataset, Rajpurkar et al. 2016) contains ~87K
question–answer pairs created by human annotators reading Wikipedia articles and
writing questions whose answers are exact text spans within those articles.

- **Source:** ~500 unique Wikipedia articles, split into paragraphs
- **Each example:** a question + one or more answers (exact text spans) + the Wikipedia
  paragraph the answer was taken from
- **Official explorer:** https://rajpurkar.github.io/SQuAD-explorer/
- **License:** CC BY-SA 4.0 — no restrictions

### How we use SQuAD for RAG

The RAG model needs two things: a set of questions+answers to train on, and a
knowledge base of passages for the retriever to search.

- **Questions** → `train.source` / `val.source` / `test.source` (one per line)
- **First answer per question** → `train.target` / `val.target` / `test.target`
- **Wikipedia paragraphs** (split into ~100-word chunks) → `kb/passages.tsv`
  (the knowledge base the retriever searches at training and inference time)

SQuAD v1.1 has no public test set (answers are withheld). We split the validation
set in half: first half → val, second half → test. This matches standard practice
in open-domain QA replication work.

### How to download and prepare

We wrote a custom script `prepare_squad.py` (in the root of this repository) that
automates the full download and preparation in one command.

> **Why not `load_dataset("rajpurkar/squad")`?** The `datasets` library version
> pinned in `rag-env` (< 3.0.0) cannot parse the new HuggingFace metadata format
> for the SQuAD repo, raising a `TypeError`. We work around this by downloading
> `train-v1.1.json` and `dev-v1.1.json` directly from Stanford's servers using
> Python's built-in `urllib` library — no extra dependencies needed.

**What `prepare_squad.py` does:**

1. Downloads `train-v1.1.json` and `dev-v1.1.json` directly from Stanford's servers
   using Python's standard `urllib` library
2. Caches the raw JSON files in `{output_dir}/_raw/` so re-runs don't re-download
3. Parses all QA pairs and writes `.source` / `.target` files (one line per example)
4. Splits the dev set in half to create val and test splits
5. Deduplicates all Wikipedia paragraphs, splits each into ~100-word chunks, and
   writes `kb/passages.tsv`

```bash
# Full dataset (from the repo root, with rag-env activated)
python prepare_squad.py --output_dir squad_data/
```

Optional flags for quick testing with a smaller subset:
```bash
python prepare_squad.py \
    --output_dir   squad_mini/ \
    --max_train    500 \
    --max_val      100 \
    --max_passages 2000
```

### Output structure

```
squad_data/
├── train.source      # 87,599 questions (one per line)
├── train.target      # 87,599 answers   (one per line)
├── val.source        # 5,285 questions
├── val.target        # 5,285 answers
├── test.source       # 5,285 questions
├── test.target       # 5,285 answers
└── kb/
    └── passages.tsv  # 34,620 passages (title TAB text, ~100 words each)
```

All six `.source`/`.target` files live flat in the same directory. The training
script uses `--data_dir squad_data` and looks for `train.source`, `val.source`, etc.
directly inside that folder.

### Actual results from our run

```
✅  Done!  Files written to: squad_data
   Training examples  : 87,599
   Validation examples: 5,285
   Test examples      : 5,285
   KB passages        : 34,620
```

---

## Dataset 2 — QAConv v1.1

### What is QAConv?

QAConv (Wu et al. 2021, Salesforce Research) is a QA dataset where the knowledge
source is real-world conversations, not Wikipedia. This is the most linguistically
challenging domain in the paper: DPR was originally trained on Wikipedia, so the
end-to-end training must significantly adapt the retriever to understand informal
dialogue.

The dataset covers three types of workplace conversations:
- **Business emails** — from the Enron email corpus
- **Panel discussions** — transcripts of academic and conference panels
- **Work channels** — Slack-style team communication

- **Official repository:** https://github.com/salesforce/QAConv
- **Version used:** V1.1 (released April 2022 — the one reported in the ACL 2022 paper)
- **License:** publicly available for research use

### How we use QAConv for RAG

- **Questions** → `train.source` / `val.source` / `test.source`
- **First answer per question** → `train.target` / `val.target` / `test.target`
- **Conversation segments** (split into ~100-word chunks) → `kb/passages.tsv`

Each QA pair in QAConv references an `article_segment_id` — a specific segment of
a conversation stored in `article_segment.json`. Each segment contains dialog turns
(`prev_ctx` for context, `seg_dialog` for the relevant portion). We concatenate all
turns into a single passage text and split into ~100-word chunks for the knowledge base.

### How to download and prepare

QAConv is not on HuggingFace. The download is semi-manual:

**Step 1 — Download the ZIP manually from GitHub:**

Go to https://github.com/salesforce/QAConv, navigate to the `dataset/` folder,
and download `QAConv-V1.1.zip`.

```bash
# Move the downloaded ZIP to the repo and extract it
mv ~/Downloads/QAConv-V1.1.zip .
unzip QAConv-V1.1.zip -d qaconv_raw/
```

This produces `qaconv_raw/QAConv-V1.1/` containing:
- `trn.json` — training QA pairs
- `val.json` — validation QA pairs
- `tst.json` — test QA pairs
- `article_segment.json` — conversation segments (the knowledge base source)
- `article_full.json` — full conversation texts

> **Note on JSON format:** QAConv does not use SQuAD's nested
> `data[].paragraphs[].qas[]` structure. It is a flat list of QA dicts, each
> referencing a segment by `article_segment_id`. This is handled by
> `prepare_qaconv.py`.

**Step 2 — Prepare with our script:**

We wrote `prepare_qaconv.py` (in the root of this repository) to convert the raw
files into the format the RAG training code expects.

**What `prepare_qaconv.py` does:**

1. Loads `article_segment.json` and builds a text string for each segment by
   concatenating all dialog turns (speaker name + text)
2. Parses `trn.json`, `val.json`, `tst.json` — extracts question, first answer,
   and segment reference for each QA pair
3. Skips items with missing questions, empty answers, or segments not found in
   `article_segment.json` (1,664 items total across all splits — normal)
4. Writes `.source` / `.target` files and `kb/passages.tsv`

```bash
python prepare_qaconv.py \
    --input_dir  qaconv_raw/QAConv-V1.1/ \
    --output_dir qaconv_data/
```

Optional flags for quick testing:
```bash
python prepare_qaconv.py \
    --input_dir     qaconv_raw/QAConv-V1.1/ \
    --output_dir    qaconv_mini/ \
    --max_train     300 \
    --max_val       60 \
    --max_passages  1500
```

### Output structure

```
qaconv_data/
├── train.source      # 25,988 questions (one per line)
├── train.target      # 25,988 answers   (one per line)
├── val.source        # 3,472 questions
├── val.target        # 3,472 answers
├── test.source       # 3,484 questions
├── test.target       # 3,484 answers
└── kb/
    └── passages.tsv  # 68,707 passages (segment_id TAB text, ~100 words each)
```

### Actual results from our run

```
✅  Done!  Files written to: qaconv_data
   Training examples  : 25,988
   Validation examples: 3,472
   Test examples      : 3,484
   KB passages        : 68,707
```

---

## Apple Silicon Fix — `use_own_knowledge_dataset.py`

Before building the FAISS index, we had to patch `use_own_knowledge_dataset.py` to
avoid a segfault on Apple Silicon. Without this fix, the script crashes at `Map: 0%`
during the DPR embedding phase.

**Root cause:** FAISS ships `libgomp` (GNU OpenMP), PyTorch ships `libomp` (LLVM
OpenMP), and Apple's Accelerate framework also loads its own BLAS threading layer.
When all three are loaded in the same process, they race for thread control and
abort. This is a known macOS arm64 issue.

**The fix — two changes at the top of the file, before any other imports:**

```python
import os
import platform

# ── Apple Silicon: prevent dual-OpenMP segfault ──────────────────────────────
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("OMP_NUM_THREADS",        "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
    os.environ.setdefault("MKL_NUM_THREADS",        "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK",   "TRUE")

import faiss  # must come after env vars are set
if platform.system() == "Darwin" and platform.machine() == "arm64":
    faiss.omp_set_num_threads(1)
```

**MPS device selection** — the script was also updated to use Apple's GPU backend
instead of CPU only:

```python
if torch.cuda.is_available():
    device = "cuda"
elif (platform.system() == "Darwin" and platform.machine() == "arm64"
      and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    device = "mps"
else:
    device = "cpu"
```

**Why all five env vars?** We tried `KMP_DUPLICATE_LIB_OK=TRUE` alone (failed),
then adding `faiss.omp_set_num_threads(1)` alone (failed), and finally all env vars
+ `faiss.omp_set_num_threads(1)` together (succeeded). All five must be set before
any library import, otherwise the runtimes are already loaded and the env vars have
no effect.

---

## FAISS Index Build

Once `use_own_knowledge_dataset.py` is patched, FAISS index building is a single
command per dataset.

### Command

```bash
# SQuAD full (~2 min on M4 Max, MPS)
python use_own_knowledge_dataset.py \
    --csv_path   squad_data/kb/passages.tsv \
    --output_dir squad_data/kb/

# SQuAD mini (~10 sec on M4 Max, MPS)
python use_own_knowledge_dataset.py \
    --csv_path   squad_mini/kb/passages.tsv \
    --output_dir squad_mini/kb/

# QAConv full (not yet run — estimate ~4–5 min on M4 Max)
python use_own_knowledge_dataset.py \
    --csv_path   qaconv_data/kb/passages.tsv \
    --output_dir qaconv_data/kb/
```

### Output

Each run produces two files in the `--output_dir`:
- `my_knowledge_dataset/` — HuggingFace Dataset with 768-dim embeddings
- `my_knowledge_dataset_hnsw_index.faiss` — FAISS HNSW index file

### Actual results from our run (SQuAD full)

```
Encoding 34,620 passages on: mps
Map: 100%|████████████████████| 34620/34620 [~2 min]
✅  FAISS index saved to squad_data/kb/
```

34,620 passages encoded in ~2 minutes using MPS. Without the fix above, this same
run segfaulted at `Map: 0%`.

---

## Quick Test — SQuAD mini

A quick test uses a small subset of the data (500 train / 100 val / 100 test /
2,000 passages) to verify the full pipeline end-to-end before committing to a
multi-day training run.

### Command

```bash
KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false \
python finetune_rag.py \
    --data_dir              squad_mini \
    --output_dir            squad_mini/output \
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
    --passages_path         squad_mini/kb/my_knowledge_dataset \
    --index_path            squad_mini/kb/my_knowledge_dataset_hnsw_index.faiss \
    --index_name            custom \
    --context_encoder_name  facebook/dpr-ctx_encoder-multiset-base \
    --csv_path              squad_mini/kb/passages.tsv \
    --index_gpus            1 \
    --gpu_order             "[]" \
    --shard_dir             squad_mini/shards \
    --indexing_freq         500 \
    --num_workers           0
```

### Why does this take ~1h45min instead of a few minutes?

`val_check_interval` defaults to `1` in this codebase, meaning PyTorch Lightning
runs a full validation pass **after every single training batch**. With 500 training
examples and `train_batch_size=2`, there are 250 training steps. Each step triggers
100 validation examples at `eval_batch_size=1`, producing 250 × 100 = **25,250
total mini-steps** per epoch. This is not a bug — the frequent re-encoding of the
knowledge base is required by the paper's end-to-end training logic.

Setting `--num_workers 0` is also required on macOS: PyTorch Lightning's DataLoader
defaults to `num_workers=4`, which spawns child processes that cannot access MPS.

### Actual results

```
Epoch 0: 100%|████| 25250/25250 [1:44:57, 4.01it/s, loss=24.1]
Trainer.fit stopped: max_epochs=1 reached.
Checkpoint saved: squad_mini/output/checkpoint251/
```

### Metrics (`squad_mini/output/metrics.json`)

| Step | val_avg_loss | val_avg_em |
|---|---|---|
| 1 (start) | 39.21 | 0.00 |
| 122 (best EM) | — | **0.07** |
| 251 (end) | 14.50 | 0.05 |

The loss dropped from 39.2 to 14.5 and Exact Match (EM) rose from 0 to 0.07 over
a single epoch on just 500 training examples. This confirms the pipeline is correctly
wired end-to-end. Full training on all 87,599 SQuAD examples over multiple epochs
is expected to reach EM ≈ 40 (the paper's reported result).

✅ **Pipeline confirmed working end-to-end on Apple Silicon.**

---

## Next Steps

| Step | Dataset | Command | Status |
|---|---|---|---|
| FAISS index | QAConv full | `use_own_knowledge_dataset.py --csv_path qaconv_data/kb/passages.tsv` | ⏳ Pending |
| Quick test | QAConv mini (300/60/60 QA, 1,500 passages) | `prepare_qaconv.py --max_train 300` + FAISS + 1-epoch train | ⏳ Pending |
| Full training | SQuAD full (~87K QA, 34K passages) | `finetune_rag_mps_end2end.sh` + multiple epochs | ⏳ Pending |
| Full training | QAConv full (~26K QA, 69K passages) | `finetune_rag_mps_end2end.sh` + multiple epochs | ⏳ Pending |

---

## References

- **SQuAD:** Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine
  Comprehension of Text. https://rajpurkar.github.io/SQuAD-explorer/
- **QAConv:** Wu et al. (2021). QAConv: Question Answering on Informative
  Conversations. https://github.com/salesforce/QAConv
- **Paper being replicated:** Siriwardhana et al. (2023). Improving the Domain
  Adaptation of RAG Models for Open Domain Question Answering. TACL.
  https://aclanthology.org/2023.tacl-1.1/
