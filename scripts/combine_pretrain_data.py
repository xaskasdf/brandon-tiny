"""Combine multiple pretrain data sources into a single tokenized dataset.

Creates a balanced mix from multiple sources (Wikipedia, SmolLM, Synthetic, etc.),
tokenized and saved as memory-mapped numpy arrays.

Usage:
    python scripts/combine_pretrain_data.py [--max_tokens MAX] [--ratios ...]
"""
import sys
import argparse
import random
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tokenizer import Tokenizer


def read_documents(filepath, max_docs=None):
    """Read text file and split into documents (separated by blank lines)."""
    docs = []
    current = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip() == '':
                if current:
                    docs.append('\n'.join(current))
                    current = []
                    if max_docs and len(docs) >= max_docs:
                        break
            else:
                current.append(line.rstrip())
    if current:
        docs.append('\n'.join(current))
    return docs


def read_documents_multi(filepaths):
    """Read documents from multiple text files."""
    all_docs = []
    for fp in filepaths:
        fp = Path(fp)
        if fp.exists():
            docs = read_documents(fp)
            print(f"    {fp.name}: {len(docs):,} docs")
            all_docs.extend(docs)
        else:
            print(f"    {fp.name}: not found, skipping")
    return all_docs


def read_parquet_documents(parquet_dir, text_column='text', max_docs=None):
    """Read documents from parquet files in a directory."""
    import pyarrow.parquet as pq

    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob('*.parquet'))
    print(f"    Found {len(files)} parquet files")

    docs = []
    for fp in files:
        table = pq.read_table(fp, columns=[text_column])
        texts = table[text_column].to_pylist()
        docs.extend(texts)
        print(f"    {fp.name}: {len(texts):,} docs (total: {len(docs):,})")
        if max_docs and len(docs) >= max_docs:
            docs = docs[:max_docs]
            break
    return docs


def tokenize_with_limit(tokenizer, docs, max_tokens):
    """Tokenize documents into a pre-allocated numpy array (memory efficient)."""
    arr = np.zeros(max_tokens, dtype=np.uint16)
    pos = 0
    for i, doc in enumerate(docs):
        tokens = tokenizer.encode(doc)
        n = len(tokens)
        if pos + n > max_tokens:
            remaining = max_tokens - pos
            if remaining > 50:
                arr[pos:pos + remaining] = tokens[:remaining]
                pos += remaining
            break
        arr[pos:pos + n] = tokens
        pos += n
        if (i + 1) % 50000 == 0:
            print(f"    Tokenized {i+1:,} docs, {pos:,} tokens ({pos/max_tokens*100:.1f}%)")
    return arr[:pos]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_tokens', type=int, default=600_000_000,
                        help='Max total tokens (default: 600M)')
    parser.add_argument('--wikipedia_ratio', type=float, default=0.0,
                        help='Fraction of tokens from Wikipedia EN')
    parser.add_argument('--tinystories_ratio', type=float, default=0.35,
                        help='Fraction of tokens from TinyStories')
    parser.add_argument('--smollm_ratio', type=float, default=0.45,
                        help='Fraction of tokens from SmolLM')
    parser.add_argument('--synthetic_ratio', type=float, default=0.20,
                        help='Fraction of tokens from synthetic data')
    parser.add_argument('--output_dir', type=str, default='data/combined',
                        help='Output directory')
    parser.add_argument('--val_tokens', type=int, default=2_000_000,
                        help='Tokens for validation set')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = Tokenizer('data/tokenizer_8k.model')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic data: all available files (auto-detect)
    synthetic_dir = Path('data/synthetic_pretrain')
    synthetic_files = sorted([
        str(f) for f in synthetic_dir.glob('train*.txt')
        if 'token' not in f.name and f.name != 'test_sample.txt'
    ])
    print(f"Found {len(synthetic_files)} synthetic files")

    # Define all sources
    sources = [
        {
            'name': 'wikipedia',
            'type': 'parquet',
            'path': 'D:/AI/datasets/wikipedia/20231101.en',
            'ratio': args.wikipedia_ratio,
        },
        {
            'name': 'tinystories',
            'type': 'text',
            'files': ['data/tinystories/train.txt'],
            'ratio': args.tinystories_ratio,
        },
        {
            'name': 'smollm',
            'type': 'text',
            'files': ['data/smollm/train.txt'],
            'ratio': args.smollm_ratio,
        },
        {
            'name': 'synthetic',
            'type': 'text',
            'files': synthetic_files,
            'ratio': args.synthetic_ratio,
        },
    ]

    ratios_str = ", ".join(f"{s['name']}={s['ratio']}" for s in sources if s['ratio'] > 0)
    print(f"Target: {args.max_tokens:,} tokens")
    print(f"Ratios: {ratios_str}")

    source_arrays = []

    for src in sources:
        name = src['name']
        ratio = src['ratio']
        target_tokens = int(args.max_tokens * ratio)

        if ratio == 0 or target_tokens == 0:
            print(f"\n--- {name} --- SKIPPED (ratio=0)")
            continue

        print(f"\n--- {name} ---")
        print(f"  Target: {target_tokens:,} tokens ({ratio*100:.0f}%)")

        # Read documents based on source type
        if src['type'] == 'parquet':
            print(f"  Reading parquet from {src['path']}...")
            docs = read_parquet_documents(src['path'])
        elif len(src['files']) == 1:
            filepath = Path(src['files'][0])
            if not filepath.exists():
                print(f"  Skipping {name}: {filepath} not found")
                continue
            print(f"  Reading {filepath}...")
            docs = read_documents(filepath)
        else:
            print(f"  Reading {len(src['files'])} files...")
            docs = read_documents_multi(src['files'])

        random.shuffle(docs)
        print(f"  Total documents: {len(docs):,}")

        tokens_arr = tokenize_with_limit(tokenizer, docs, target_tokens)
        print(f"  Tokenized: {len(tokens_arr):,} tokens ({tokens_arr.nbytes / 1e6:.1f} MB)")
        source_arrays.append(tokens_arr)
        del docs  # Free memory

    # Concatenate all sources
    print("\nConcatenating sources...")
    all_tokens = np.concatenate(source_arrays)
    del source_arrays

    # Shuffle at chunk level (not token level) to mix sources
    chunk_size = 512
    n_full_chunks = len(all_tokens) // chunk_size
    remainder = all_tokens[n_full_chunks * chunk_size:]
    chunks = all_tokens[:n_full_chunks * chunk_size].reshape(-1, chunk_size)
    print(f"Shuffling {n_full_chunks:,} chunks...")
    np.random.shuffle(chunks)
    all_tokens = np.concatenate([chunks.ravel(), remainder])
    del chunks, remainder

    total = len(all_tokens)
    print(f"\n=== Total: {total:,} tokens ({all_tokens.nbytes / 1e9:.2f} GB) ===")

    # Split into train/val
    val_size = min(args.val_tokens, total // 10)
    train_arr = all_tokens[:-val_size]
    val_arr = all_tokens[-val_size:]

    print(f"  Train: {len(train_arr):,} tokens")
    print(f"  Val: {len(val_arr):,} tokens")

    train_path = output_dir / 'train_tokens.npy'
    val_path = output_dir / 'validation_tokens.npy'

    np.save(train_path, train_arr)
    np.save(val_path, val_arr)

    print(f"\n  Saved: {train_path} ({train_arr.nbytes / 1e6:.1f} MB)")
    print(f"  Saved: {val_path} ({val_arr.nbytes / 1e6:.1f} MB)")
    print("\nDone!")


if __name__ == '__main__':
    main()
