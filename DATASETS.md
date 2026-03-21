# Dataset Setup Guide

This document tracks the dataset preparation process for the replication of
Siriwardhana et al. (TACL 2023). It is written so that anyone — including future
readers of this repository — can reproduce every step from scratch.

> **Status:** ✅ Datasets downloaded and prepared — FAISS index pending.

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

**What `prepare_squad.py` does:**
1. Downloads `train-v1.1.json` and `dev-v1.1.json` directly from Stanford's servers
   using Python's standard `urllib` library (no extra dependencies needed)
2. Caches the raw JSON files in `{output_dir}/_raw/` so re-runs don't re-download
3. Parses all QA pairs and writes `.source` / `.target` files (one line per example)
4. Splits the dev set in half to create val and test splits
5. Deduplicates all Wikipedia paragraphs across both splits, splits each into
   ~100-word chunks, and writes `kb/passages.tsv`

```bash
# From the repo root, with rag-env activated
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
- `convert_txt.py` — Salesforce's own conversion helper

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

## Next steps (not yet executed)

The following steps are documented here as reference and will be updated as they
are completed.

### Build the FAISS knowledge base indexes

Before training, each dataset's `passages.tsv` must be encoded into dense vectors
using the DPR context encoder and indexed with FAISS. This is done with the existing
`use_own_knowledge_dataset.py` script.

```bash
# SQuAD (~30–40 min on M4 Max)
python use_own_knowledge_dataset.py \
    --csv_path   squad_data/kb/passages.tsv \
    --output_dir squad_data/kb/

# QAConv (~60–90 min on M4 Max, larger KB)
python use_own_knowledge_dataset.py \
    --csv_path   qaconv_data/kb/passages.tsv \
    --output_dir qaconv_data/kb/
```

### Run quick validation tests (≤ 30 min each)

Before committing to multi-day training runs, verify the full pipeline is correctly
wired using small subsets. See `DATASETS.md` for commands — this section will be
updated with actual results once the tests are run.

### Run full training

Full training commands and expected metrics will be documented here once the quick
tests confirm the pipeline is working correctly.

---

## References

- **SQuAD:** Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine
  Comprehension of Text. https://rajpurkar.github.io/SQuAD-explorer/
- **QAConv:** Wu et al. (2021). QAConv: Question Answering on Informative
  Conversations. https://github.com/salesforce/QAConv
- **Paper being replicated:** Siriwardhana et al. (2023). Improving the Domain
  Adaptation of RAG Models for Open Domain Question Answering. TACL.
  https://aclanthology.org/2023.tacl-1.1/
