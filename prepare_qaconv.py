#!/usr/bin/env python3
"""
prepare_qaconv.py — Prepare the QAConv dataset for RAG end-to-end training.

Reads the extracted QAConv-V1.1 directory and writes all output to --output_dir:

    {output_dir}/train.source     one question per line  (~27,287 lines)
    {output_dir}/train.target     one answer per line    (~27,287 lines)
    {output_dir}/val.source       validation questions
    {output_dir}/val.target       validation answers
    {output_dir}/test.source      test questions
    {output_dir}/test.target      test answers
    {output_dir}/kb/passages.tsv  knowledge base (title TAB text, 100 words each)

The knowledge base passages are built from article_segment.json: each segment's
dialog turns (prev_ctx + seg_dialog) are concatenated into a single passage text,
then split into ~100-word chunks.

Usage — full dataset:
    python prepare_qaconv.py \\
        --input_dir  qaconv_raw/QAConv-V1.1/ \\
        --output_dir qaconv_data/

Usage — quick mini test (≤ 30 min end-to-end):
    python prepare_qaconv.py \\
        --input_dir     qaconv_raw/QAConv-V1.1/ \\
        --output_dir    qaconv_mini/ \\
        --max_train     300 \\
        --max_val       60 \\
        --max_passages  1500
"""

import argparse
import json
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────────

def sanitize(s: str) -> str:
    """Remove characters that break the one-line-per-example format."""
    return str(s).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def write_lines(path: Path, lines: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(sanitize(line) + "\n")
    print(f"  Wrote {len(lines):>8,} lines  →  {path}")


def turns_to_text(turns: list) -> str:
    """
    Convert a list of dialog turns into a single string.
    Each turn is {"id": ..., "speaker": "...", "text": "..."}.
    """
    parts = []
    for turn in turns:
        speaker = turn.get("speaker", "").strip()
        text    = turn.get("text", "").strip()
        if text:
            parts.append(f"{speaker}: {text}" if speaker else text)
    return " ".join(parts)


def split_text(text: str, n: int = 100) -> list:
    """Split text into chunks of ~n whitespace-separated words."""
    words = text.split()
    return [
        " ".join(words[i : i + n]).strip()
        for i in range(0, len(words), n)
        if words[i : i + n]
    ]


def build_segment_texts(article_segment: dict) -> dict:
    """
    Build a mapping from segment_id → full passage text.

    Each segment has:
        prev_ctx   : list of dialog turns before the segment (context)
        seg_dialog : list of dialog turns in the segment     (the relevant part)

    We concatenate both for maximum context, matching how the model will need
    to find answers within the passage.
    """
    segment_texts = {}
    for seg_id, seg in article_segment.items():
        prev  = seg.get("prev_ctx",   [])
        main  = seg.get("seg_dialog", [])
        text  = turns_to_text(prev + main)
        if text:
            segment_texts[seg_id] = text
    return segment_texts


def parse_qa_file(path: Path, segment_texts: dict) -> tuple:
    """
    Parse a trn/val/tst QA file.

    Each element:
        {
          "id":               "trn-0",
          "article_segment_id": "enron_wisrlab-769",
          "article_full_id":  [...],
          "QG":               false,
          "question":         "How can someone get in touch with Puthigai?",
          "answers":          ["call me at (713) 853-3399."]
        }

    Returns:
        questions : list[str]
        answers   : list[str]   (first answer per question)
        seg_ids   : list[str]   (segment IDs referenced — for KB building)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions, answers, seg_ids = [], [], []
    skipped = 0
    for item in data:
        q   = item.get("question", "").strip()
        ans = item.get("answers", [])
        sid = item.get("article_segment_id", "")

        if not q or not ans or not ans[0].strip():
            skipped += 1
            continue
        if sid not in segment_texts:
            skipped += 1
            continue

        questions.append(q)
        answers.append(ans[0].strip())
        seg_ids.append(sid)

    if skipped:
        print(f"    (skipped {skipped} items with missing question/answer/segment)")

    return questions, answers, seg_ids


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare QAConv-V1.1 for RAG training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the extracted QAConv-V1.1 directory "
             "(should contain trn.json, val.json, tst.json, article_segment.json).",
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
        help="Maximum number of training QA pairs (default: all ~27,287).",
    )
    parser.add_argument(
        "--max_val",
        type=int,
        default=None,
        help="Maximum QA pairs per val/test split (default: all).",
    )
    parser.add_argument(
        "--max_passages",
        type=int,
        default=None,
        help="Maximum number of KB passages (default: all).",
    )
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)

    # Validate input
    required = ["trn.json", "val.json", "tst.json", "article_segment.json"]
    missing  = [f for f in required if not (inp / f).exists()]
    if missing:
        raise SystemExit(
            f"ERROR: Missing files in {inp}: {missing}\n"
            f"Make sure --input_dir points to the QAConv-V1.1 folder.\n"
            f"Contents of {inp}:\n"
            + "\n".join(f"  {p.name}" for p in sorted(inp.iterdir()))
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "kb").mkdir(exist_ok=True)

    # ── Load article segments (the knowledge base source) ─────────────────────
    print("Loading article_segment.json…")
    with open(inp / "article_segment.json", encoding="utf-8") as f:
        article_segment = json.load(f)
    print(f"  {len(article_segment):,} segments loaded")

    print("Building segment texts…")
    segment_texts = build_segment_texts(article_segment)
    print(f"  {len(segment_texts):,} non-empty segments")

    # ── Parse QA splits ───────────────────────────────────────────────────────
    print("\nParsing trn.json (training)…")
    train_q, train_a, train_sids = parse_qa_file(inp / "trn.json", segment_texts)
    print(f"  {len(train_q):,} QA pairs")

    print("Parsing val.json (validation)…")
    val_q, val_a, val_sids = parse_qa_file(inp / "val.json", segment_texts)
    print(f"  {len(val_q):,} QA pairs")

    print("Parsing tst.json (test)…")
    test_q, test_a, test_sids = parse_qa_file(inp / "tst.json", segment_texts)
    print(f"  {len(test_q):,} QA pairs")

    # ── Apply limits ──────────────────────────────────────────────────────────
    if args.max_train is not None:
        train_q, train_a = train_q[:args.max_train], train_a[:args.max_train]
        train_sids = train_sids[:args.max_train]
    if args.max_val is not None:
        val_q,  val_a  = val_q[:args.max_val],  val_a[:args.max_val]
        test_q, test_a = test_q[:args.max_val], test_a[:args.max_val]
        val_sids  = val_sids[:args.max_val]
        test_sids = test_sids[:args.max_val]

    # ── Build knowledge base from ALL referenced segments ─────────────────────
    # Include segments referenced by any split so the retriever can find answers
    # for val and test questions too.
    print("\nBuilding knowledge base passages…")
    all_sids = set(train_sids) | set(val_sids) | set(test_sids)

    kb_passages: list = []
    seen_sids:   set  = set()

    for seg_id, text in segment_texts.items():
        if args.max_passages is not None and len(kb_passages) >= args.max_passages:
            break
        if seg_id in seen_sids:
            continue
        seen_sids.add(seg_id)

        # Prioritize segments referenced by our QA splits
        # (when using --max_passages the most relevant ones come first)
        if seg_id not in all_sids and args.max_passages is not None:
            continue  # skip unreferenced segments when we have a limit

        for chunk in split_text(text, n=100):
            kb_passages.append((seg_id, chunk))
            if args.max_passages is not None and len(kb_passages) >= args.max_passages:
                break

    # If we still have room and a limit, fill with remaining segments
    if args.max_passages is None or len(kb_passages) < args.max_passages:
        for seg_id, text in segment_texts.items():
            if seg_id in seen_sids:
                continue
            seen_sids.add(seg_id)
            for chunk in split_text(text, n=100):
                kb_passages.append((seg_id, chunk))
                if args.max_passages is not None and len(kb_passages) >= args.max_passages:
                    break
            if args.max_passages is not None and len(kb_passages) >= args.max_passages:
                break

    if args.max_passages is not None:
        kb_passages = kb_passages[:args.max_passages]

    print(f"  {len(kb_passages):,} passages from {len(seen_sids):,} unique segments")

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
        for seg_id, text in kb_passages:
            f.write(f"{sanitize(seg_id)}\t{sanitize(text)}\n")
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
