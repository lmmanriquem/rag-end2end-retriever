#!/usr/bin/env python3
"""
prepare_squad.py — Prepare the SQuAD dataset for RAG end-to-end training.

Downloads the official SQuAD v1.1 JSON files directly from Stanford
(no HuggingFace datasets library required — uses only Python stdlib).

Writes all output to --output_dir:

    {output_dir}/train.source     one question per line  (~87,599 lines)
    {output_dir}/train.target     one answer per line    (~87,599 lines)
    {output_dir}/val.source       validation questions   (~5,285 lines)
    {output_dir}/val.target       validation answers     (~5,285 lines)
    {output_dir}/test.source      test questions         (~5,285 lines)
    {output_dir}/test.target      test answers           (~5,285 lines)
    {output_dir}/kb/passages.tsv  knowledge base         (~30,000 rows, title TAB text)

SQuAD has no public test set, so the validation split is divided in half:
the first half → val, the second half → test.

Usage — full dataset:
    python prepare_squad.py --output_dir squad_data/

Usage — quick mini test (≤ 30 min end-to-end):
    python prepare_squad.py --output_dir squad_mini/ \\
        --max_train 500 --max_val 100 --max_passages 2000
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Official SQuAD v1.1 files hosted by Stanford
SQUAD_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
SQUAD_DEV_URL   = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def download_json(url: str, dest: Path) -> dict:
    """Download a JSON file if not already cached, then load and return it."""
    if dest.exists():
        print(f"  Using cached file: {dest}")
    else:
        print(f"  Downloading {url} …")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            raise SystemExit(
                f"\nERROR: Could not download {url}\n"
                f"  {e}\n\n"
                "Please check your internet connection. If the Stanford URL is unavailable,\n"
                "you can also download the files manually:\n"
                f"  curl -O {url}\n"
                f"and place them at {dest}"
            )
        print(f"  Saved to {dest}")
    with open(dest, encoding="utf-8") as f:
        return json.load(f)


def parse_squad_json(data: dict) -> tuple:
    """
    Parse a SQuAD v1.1 JSON dict.

    Format:
        { "data": [ { "title": "...", "paragraphs": [
            { "context": "...", "qas": [
                { "question": "...", "answers": [{"text": "...", ...}] }
            ]}
        ]}]}

    Returns:
        questions : list[str]
        answers   : list[str]   (first answer per question)
        passages  : list[tuple[str, str]]  (title, context)  — one entry per paragraph
    """
    questions, answers, passages = [], [], []
    for article in data["data"]:
        title = article["title"]
        for para in article["paragraphs"]:
            context = para["context"].strip()
            passages.append((title, context))
            for qa in para["qas"]:
                if not qa.get("answers"):
                    continue
                questions.append(qa["question"].strip())
                answers.append(qa["answers"][0]["text"].strip())
    return questions, answers, passages


def split_text(text: str, n: int = 100) -> list:
    """Split text into chunks of ~n whitespace-separated words."""
    words = text.split()
    return [
        " ".join(words[i : i + n]).strip()
        for i in range(0, len(words), n)
        if words[i : i + n]
    ]


def sanitize(s: str) -> str:
    """Remove characters that break the one-line-per-example format."""
    return str(s).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def write_lines(path: Path, lines: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(sanitize(line) + "\n")
    print(f"  Wrote {len(lines):>8,} lines  →  {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download SQuAD v1.1 from Stanford and prepare it for RAG training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where all output files will be written.",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="Maximum number of training QA pairs (default: all ~87,599).",
    )
    parser.add_argument(
        "--max_val",
        type=int,
        default=None,
        help="Maximum QA pairs per val/test split (default: ~5,285 each).",
    )
    parser.add_argument(
        "--max_passages",
        type=int,
        default=None,
        help="Maximum number of KB passages (default: all ~30,000).",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kb").mkdir(exist_ok=True)

    # Cache the raw JSON files inside the output directory so re-runs are fast
    cache_dir = out / "_raw"
    cache_dir.mkdir(exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────────
    print("Downloading SQuAD v1.1 from Stanford…")
    train_json = download_json(SQUAD_TRAIN_URL, cache_dir / "train-v1.1.json")
    dev_json   = download_json(SQUAD_DEV_URL,   cache_dir / "dev-v1.1.json")

    # ── Parse ─────────────────────────────────────────────────────────────────
    print("\nParsing training split…")
    train_q, train_a, train_passages = parse_squad_json(train_json)
    print(f"  {len(train_q):,} QA pairs  |  {len(train_passages):,} paragraphs")

    print("Parsing validation split…")
    dev_q, dev_a, dev_passages = parse_squad_json(dev_json)
    print(f"  {len(dev_q):,} QA pairs  |  {len(dev_passages):,} paragraphs")

    # ── Apply limits ──────────────────────────────────────────────────────────
    if args.max_train is not None:
        train_q = train_q[: args.max_train]
        train_a = train_a[: args.max_train]

    # SQuAD has no public test set → split dev in half
    mid = len(dev_q) // 2
    val_q,  val_a  = dev_q[:mid],  dev_a[:mid]
    test_q, test_a = dev_q[mid:],  dev_a[mid:]

    if args.max_val is not None:
        val_q,  val_a  = val_q[: args.max_val],  val_a[: args.max_val]
        test_q, test_a = test_q[: args.max_val], test_a[: args.max_val]

    # ── Knowledge base: unique contexts from BOTH splits ─────────────────────
    # Each (title, context) paragraph is split into ~100-word chunks.
    # passages.tsv format: title TAB text  (NO header — use_own_knowledge_dataset.py
    # reads it with column_names=["title","text"])
    print("\nBuilding knowledge base passages…")
    seen_contexts: set = set()
    kb_passages: list = []

    for title, context in train_passages + dev_passages:
        if context in seen_contexts:
            continue
        seen_contexts.add(context)
        for chunk in split_text(context, n=100):
            kb_passages.append((title, chunk))
        if args.max_passages is not None and len(kb_passages) >= args.max_passages:
            break

    if args.max_passages is not None:
        kb_passages = kb_passages[: args.max_passages]

    print(f"  {len(kb_passages):,} passages from {len(seen_contexts):,} unique paragraphs")

    # ── Write QA split files ──────────────────────────────────────────────────
    print("\nWriting QA split files…")
    write_lines(out / "train.source", train_q)
    write_lines(out / "train.target", train_a)
    write_lines(out / "val.source",   val_q)
    write_lines(out / "val.target",   val_a)
    write_lines(out / "test.source",  test_q)
    write_lines(out / "test.target",  test_a)

    # ── Write knowledge base ──────────────────────────────────────────────────
    print("\nWriting knowledge base (passages.tsv)…")
    tsv_path = out / "kb" / "passages.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        for title, text in kb_passages:
            f.write(f"{sanitize(title)}\t{sanitize(text)}\n")
    print(f"  Wrote {len(kb_passages):>8,} passages  →  {tsv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅  Done!  Files written to:", out)
    print(f"   Training examples  : {len(train_q):,}")
    print(f"   Validation examples: {len(val_q):,}")
    print(f"   Test examples      : {len(test_q):,}")
    print(f"   KB passages        : {len(kb_passages):,}")
    print()
    print("Next step — build the FAISS index:")
    print(f"   python use_own_knowledge_dataset.py \\")
    print(f"       --csv_path   {tsv_path} \\")
    print(f"       --output_dir {out / 'kb'}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
