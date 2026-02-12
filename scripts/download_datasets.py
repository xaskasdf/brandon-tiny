#!/usr/bin/env python
"""
Download and prepare datasets from HuggingFace for TinyLlama training.

Datasets:
  Pre-training:
    - TinyStories (roneneldan/TinyStories) - short stories for small LMs (<10M)
    - FineWeb-Edu (HuggingFaceFW/fineweb-edu-score-2) - educational web (10M-500M)
    - Cosmopedia (HuggingFaceTB/cosmopedia) - synthetic textbooks (10M-500M)

  Instruction tuning:
    - Alpaca (tatsu-lab/alpaca) - 52K instruction/output pairs
    - Dolly (databricks/databricks-dolly-15k) - 15K instructions

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --tinystories
    python scripts/download_datasets.py --fineweb --max-tokens 500000000
    python scripts/download_datasets.py --alpaca --dolly --merge
"""

import argparse
import json
from pathlib import Path


def download_tinystories(output_dir: str = "data/tinystories"):
    """Download TinyStories dataset for pre-training."""
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories")

    # Save train split
    train_file = out / "train.txt"
    print(f"Writing train split ({len(ds['train'])} stories)...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(ds['train']):
            text = example.get('text', '')
            if text.strip():
                f.write(text.strip() + '\n\n')
            if (i + 1) % 100000 == 0:
                print(f"  {i + 1:,} stories written...")

    # Save validation split
    val_file = out / "validation.txt"
    print(f"Writing validation split ({len(ds['validation'])} stories)...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in ds['validation']:
            text = example.get('text', '')
            if text.strip():
                f.write(text.strip() + '\n\n')

    # Stats
    train_size = train_file.stat().st_size / (1024 * 1024)
    val_size = val_file.stat().st_size / (1024 * 1024)
    print(f"\nTinyStories saved:")
    print(f"  Train: {train_file} ({train_size:.1f} MB)")
    print(f"  Val:   {val_file} ({val_size:.1f} MB)")


def download_alpaca(output_dir: str = "data/chat"):
    """Download Alpaca dataset for instruction tuning."""
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading Alpaca (52K instructions)...")
    ds = load_dataset("tatsu-lab/alpaca")

    # Convert to our JSONL format
    train_file = out / "alpaca.jsonl"
    count = 0
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in ds['train']:
            instruction = example.get('instruction', '').strip()
            inp = example.get('input', '').strip()
            output = example.get('output', '').strip()

            if not instruction or not output:
                continue

            # Combine instruction + input if present
            if inp:
                full_instruction = f"{instruction}\n{inp}"
            else:
                full_instruction = instruction

            record = {
                "instruction": full_instruction,
                "input": "",
                "output": output
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

    print(f"Alpaca saved: {train_file} ({count:,} examples)")


def download_dolly(output_dir: str = "data/chat"):
    """Download Dolly dataset for instruction tuning."""
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading Dolly (15K instructions)...")
    ds = load_dataset("databricks/databricks-dolly-15k")

    train_file = out / "dolly.jsonl"
    count = 0
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in ds['train']:
            instruction = example.get('instruction', '').strip()
            context = example.get('context', '').strip()
            response = example.get('response', '').strip()

            if not instruction or not response:
                continue

            if context:
                full_instruction = f"{instruction}\nContext: {context}"
            else:
                full_instruction = instruction

            record = {
                "instruction": full_instruction,
                "input": "",
                "output": response
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

    print(f"Dolly saved: {train_file} ({count:,} examples)")


def download_smollm_corpus(output_dir: str = "data/smollm", max_tokens: int = 1_000_000_000):
    """Download SmolLM-Corpus (Cosmopedia v2) for pre-training 10M-100M models.

    The corpus used to train SmolLM-135M/360M. High-quality synthetic educational content.
    Default: 1B tokens, good for 10M-30M models.
    """
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SmolLM-Corpus/Cosmopedia-v2 (target: {max_tokens/1e6:.0f}M tokens)...")
    print("Using streaming mode...")

    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True
    )

    train_file = out / "train.txt"
    val_file = out / "validation.txt"

    chars_per_token = 4
    max_chars = max_tokens * chars_per_token
    val_chars = int(max_chars * 0.05)
    train_chars = max_chars - val_chars

    total_chars = 0
    doc_count = 0

    print("Writing train split...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in ds:
            text = example.get('text', '')
            if not text or len(text) < 50:
                continue

            f.write(text.strip() + '\n\n')
            total_chars += len(text)
            doc_count += 1

            if doc_count % 10000 == 0:
                est_tokens = total_chars / chars_per_token
                print(f"  {doc_count:,} docs, ~{est_tokens/1e6:.1f}M tokens")

            if total_chars >= train_chars:
                break

    print("Writing validation split...")
    val_count = 0
    val_total = 0
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in ds:
            text = example.get('text', '')
            if not text or len(text) < 50:
                continue

            f.write(text.strip() + '\n\n')
            val_total += len(text)
            val_count += 1

            if val_total >= val_chars:
                break

    total_chars += val_total
    train_size = train_file.stat().st_size / (1024 * 1024)
    val_size = val_file.stat().st_size / (1024 * 1024)
    est_tokens = total_chars / chars_per_token

    print(f"\nSmolLM-Corpus saved:")
    print(f"  Train: {train_file} ({train_size:.1f} MB, {doc_count:,} docs)")
    print(f"  Val:   {val_file} ({val_size:.1f} MB, {val_count:,} docs)")
    print(f"  Est. tokens: ~{est_tokens/1e6:.0f}M")


def download_minipile(output_dir: str = "data/minipile"):
    """Download MiniPile - curated 6GB subset of The Pile.

    22 diverse domains (Wikipedia, books, code, ArXiv, etc).
    ~1.5B tokens. Good for <10M and 10M-30M models.
    """
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading MiniPile (~1.5B tokens, 6GB text)...")

    ds = load_dataset("JeanKaddour/minipile")

    for split_name, out_name in [("train", "train"), ("validation", "validation"), ("test", "test")]:
        split_file = out / f"{out_name}.txt"
        split_data = ds[split_name]
        print(f"Writing {split_name} split ({len(split_data)} docs)...")
        with open(split_file, 'w', encoding='utf-8') as f:
            for i, example in enumerate(split_data):
                text = example.get('text', '')
                if text.strip():
                    f.write(text.strip() + '\n\n')
                if (i + 1) % 100000 == 0:
                    print(f"  {i + 1:,} docs written...")

        size = split_file.stat().st_size / (1024 * 1024)
        print(f"  {split_file} ({size:.1f} MB)")

    print(f"\nMiniPile saved to {out}")


def download_fineweb_edu(output_dir: str = "data/fineweb", max_tokens: int = 500_000_000):
    """Download FineWeb-Edu subset for pre-training medium models (10M-500M).

    Uses streaming to avoid downloading the full 220B token dataset.
    Default: 500M tokens (~2GB text), good for 30M-110M models.
    """
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FineWeb-Edu (target: {max_tokens/1e6:.0f}M tokens)...")
    print("Using streaming mode to avoid full download...")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu-score-2",
        split="train",
        streaming=True
    )

    train_file = out / "train.txt"
    val_file = out / "validation.txt"

    # Estimate ~4 chars per token
    chars_per_token = 4
    max_chars = max_tokens * chars_per_token
    val_chars = int(max_chars * 0.05)  # 5% for validation
    train_chars = max_chars - val_chars

    total_chars = 0
    doc_count = 0

    # Write train split
    print("Writing train split...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in ds:
            text = example.get('text', '')
            if not text or len(text) < 50:
                continue

            f.write(text.strip() + '\n\n')
            total_chars += len(text)
            doc_count += 1

            if doc_count % 10000 == 0:
                est_tokens = total_chars / chars_per_token
                print(f"  {doc_count:,} docs, ~{est_tokens/1e6:.1f}M tokens")

            if total_chars >= train_chars:
                break

    # Write val split (continue from same stream)
    print("Writing validation split...")
    val_count = 0
    val_total = 0
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in ds:
            text = example.get('text', '')
            if not text or len(text) < 50:
                continue

            f.write(text.strip() + '\n\n')
            val_total += len(text)
            val_count += 1

            if val_total >= val_chars:
                break

    total_chars += val_total
    train_size = train_file.stat().st_size / (1024 * 1024)
    val_size = val_file.stat().st_size / (1024 * 1024)
    est_tokens = total_chars / chars_per_token

    print(f"\nFineWeb-Edu saved:")
    print(f"  Train: {train_file} ({train_size:.1f} MB, {doc_count:,} docs)")
    print(f"  Val:   {val_file} ({val_size:.1f} MB, {val_count:,} docs)")
    print(f"  Est. tokens: ~{est_tokens/1e6:.0f}M")


def download_cosmopedia(output_dir: str = "data/cosmopedia", max_tokens: int = 200_000_000):
    """Download Cosmopedia subset - synthetic textbooks for pre-training.

    Great for models 10M-500M. Content is educational and structured.
    Default: 200M tokens (~800MB text).
    """
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Cosmopedia (target: {max_tokens/1e6:.0f}M tokens)...")
    print("Using streaming mode...")

    ds = load_dataset(
        "HuggingFaceTB/cosmopedia",
        split="train",
        streaming=True
    )

    train_file = out / "train.txt"
    val_file = out / "validation.txt"

    chars_per_token = 4
    max_chars = max_tokens * chars_per_token
    val_chars = int(max_chars * 0.05)
    train_chars = max_chars - val_chars

    total_chars = 0
    doc_count = 0

    print("Writing train split...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in ds:
            text = example.get('text', '')
            if not text or len(text) < 50:
                continue

            f.write(text.strip() + '\n\n')
            total_chars += len(text)
            doc_count += 1

            if doc_count % 10000 == 0:
                est_tokens = total_chars / chars_per_token
                print(f"  {doc_count:,} docs, ~{est_tokens/1e6:.1f}M tokens")

            if total_chars >= train_chars:
                break

    print("Writing validation split...")
    val_count = 0
    val_total = 0
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in ds:
            text = example.get('text', '')
            if not text or len(text) < 50:
                continue

            f.write(text.strip() + '\n\n')
            val_total += len(text)
            val_count += 1

            if val_total >= val_chars:
                break

    total_chars += val_total
    train_size = train_file.stat().st_size / (1024 * 1024)
    val_size = val_file.stat().st_size / (1024 * 1024)
    est_tokens = total_chars / chars_per_token

    print(f"\nCosmopedia saved:")
    print(f"  Train: {train_file} ({train_size:.1f} MB, {doc_count:,} docs)")
    print(f"  Val:   {val_file} ({val_size:.1f} MB, {val_count:,} docs)")
    print(f"  Est. tokens: ~{est_tokens/1e6:.0f}M")


def merge_instruction_data(output_dir: str = "data/chat"):
    """Merge all instruction datasets into train/val splits."""
    out = Path(output_dir)
    all_examples = []

    # Load all JSONL files
    for jsonl_file in sorted(out.glob("*.jsonl")):
        if jsonl_file.name in ('train.jsonl', 'val.jsonl'):
            continue
        print(f"Loading {jsonl_file.name}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_examples.append(json.loads(line))

    if not all_examples:
        print("No instruction data found to merge.")
        return

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_examples)

    # Split 90/10
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # Write
    train_file = out / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    val_file = out / "val.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"\nMerged instruction data:")
    print(f"  Sources: {[f.name for f in sorted(out.glob('*.jsonl')) if f.name not in ('train.jsonl', 'val.jsonl')]}")
    print(f"  Train: {len(train_examples):,} examples")
    print(f"  Val:   {len(val_examples):,} examples")


def main():
    parser = argparse.ArgumentParser(description="Download datasets from HuggingFace")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--tinystories", action="store_true", help="Download TinyStories (<10M models)")
    parser.add_argument("--minipile", action="store_true", help="Download MiniPile (<10M, 10-30M models)")
    parser.add_argument("--smollm", action="store_true", help="Download SmolLM-Corpus/Cosmopedia-v2 (10-100M)")
    parser.add_argument("--fineweb", action="store_true", help="Download FineWeb-Edu (30M-500M)")
    parser.add_argument("--cosmopedia", action="store_true", help="Download Cosmopedia v1 (10-500M)")
    parser.add_argument("--alpaca", action="store_true", help="Download Alpaca instructions")
    parser.add_argument("--dolly", action="store_true", help="Download Dolly instructions")
    parser.add_argument("--merge", action="store_true", help="Merge instruction datasets")
    parser.add_argument("--max-tokens", type=int, default=500_000_000,
                        help="Max tokens for streaming datasets (default: 500M)")
    args = parser.parse_args()

    if args.all:
        args.tinystories = True
        args.minipile = True
        args.smollm = True
        args.fineweb = True
        args.cosmopedia = True
        args.alpaca = True
        args.dolly = True
        args.merge = True

    if not any([args.tinystories, args.minipile, args.smollm, args.fineweb,
                args.cosmopedia, args.alpaca, args.dolly, args.merge]):
        parser.print_help()
        return

    if args.tinystories:
        download_tinystories()
        print()

    if args.minipile:
        download_minipile()
        print()

    if args.smollm:
        download_smollm_corpus(max_tokens=args.max_tokens)
        print()

    if args.fineweb:
        download_fineweb_edu(max_tokens=args.max_tokens)
        print()

    if args.cosmopedia:
        download_cosmopedia(max_tokens=min(args.max_tokens, 200_000_000))
        print()

    if args.alpaca:
        download_alpaca()
        print()

    if args.dolly:
        download_dolly()
        print()

    if args.merge:
        merge_instruction_data()

    print("\nDone!")


if __name__ == "__main__":
    main()
