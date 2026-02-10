"""
Data preprocessing utilities for TinyLlama.

Handles:
- Wikipedia parquet to text conversion
- Wikitext preparation
- Data sharding for distributed training
"""

import os
import json
from pathlib import Path
from typing import Optional, Iterator
from tqdm import tqdm


def process_wikipedia_parquet(
    input_dir: str,
    output_file: str,
    max_articles: Optional[int] = None,
    min_length: int = 100
) -> int:
    """
    Process Wikipedia parquet files to plain text.

    Args:
        input_dir: Directory containing parquet files
        output_file: Output text file path
        max_articles: Maximum number of articles to process
        min_length: Minimum article length in characters

    Returns:
        Number of articles processed
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Please install pyarrow: pip install pyarrow")

    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = list(input_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    article_count = 0
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for pq_file in tqdm(parquet_files, desc="Processing parquet files"):
            table = pq.read_table(pq_file)
            df = table.to_pandas()

            # Wikipedia dump typically has 'text' column
            text_col = 'text' if 'text' in df.columns else df.columns[0]

            for _, row in df.iterrows():
                text = str(row[text_col])

                # Filter short articles
                if len(text) < min_length:
                    continue

                # Write article with separator
                out_f.write(text.strip())
                out_f.write('\n\n')

                article_count += 1
                if max_articles and article_count >= max_articles:
                    break

            if max_articles and article_count >= max_articles:
                break

    print(f"Processed {article_count:,} articles to {output_file}")
    return article_count


def prepare_wikitext(
    output_dir: str,
    version: str = "wikitext-2-raw-v1"
) -> dict:
    """
    Download and prepare Wikitext dataset.

    Args:
        output_dir: Output directory
        version: Dataset version ("wikitext-2-raw-v1" or "wikitext-103-raw-v1")

    Returns:
        Dict with paths to train/val/test files
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {version}...")
    dataset = load_dataset("wikitext", version)

    paths = {}
    for split in ['train', 'validation', 'test']:
        split_data = dataset[split]
        output_file = output_path / f"{split}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in split_data:
                text = example['text']
                if text.strip():
                    f.write(text)
                    f.write('\n')

        paths[split] = str(output_file)
        print(f"Saved {split}: {len(split_data)} examples")

    return paths


def shard_tokenized_data(
    input_file: str,
    output_dir: str,
    tokenizer,
    shard_size: int = 100_000_000,
    split: str = "train"
) -> int:
    """
    Tokenize text file and save as sharded binary files.

    Args:
        input_file: Input text file
        output_dir: Output directory for shards
        tokenizer: Tokenizer instance
        shard_size: Tokens per shard
        split: Split name for output files

    Returns:
        Total number of tokens
    """
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token_buffer = []
    shard_idx = 0
    total_tokens = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Tokenizing"):
            if line.strip():
                tokens = tokenizer.encode(line)
                token_buffer.extend(tokens)
                total_tokens += len(tokens)

                # Save shard when buffer is full
                while len(token_buffer) >= shard_size:
                    shard_tokens = token_buffer[:shard_size]
                    token_buffer = token_buffer[shard_size:]

                    shard_file = output_path / f"{split}_{shard_idx:04d}.bin"
                    arr = np.array(shard_tokens, dtype=np.uint16)
                    arr.tofile(shard_file)
                    print(f"Saved shard {shard_idx}: {len(shard_tokens):,} tokens")
                    shard_idx += 1

    # Save remaining tokens
    if token_buffer:
        if shard_idx == 0:
            shard_file = output_path / f"{split}.bin"
        else:
            shard_file = output_path / f"{split}_{shard_idx:04d}.bin"

        arr = np.array(token_buffer, dtype=np.uint16)
        arr.tofile(shard_file)
        print(f"Saved final shard: {len(token_buffer):,} tokens")

    print(f"Total: {total_tokens:,} tokens in {shard_idx + 1} shards")
    return total_tokens


def create_train_val_split(
    input_file: str,
    output_dir: str,
    val_ratio: float = 0.01,
    seed: int = 42
) -> dict:
    """
    Split a text file into train and validation sets.

    Args:
        input_file: Input text file
        output_dir: Output directory
        val_ratio: Fraction of data for validation
        seed: Random seed

    Returns:
        Dict with paths to train and val files
    """
    import random
    random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Shuffle and split
    random.shuffle(lines)
    val_size = int(len(lines) * val_ratio)

    val_lines = lines[:val_size]
    train_lines = lines[val_size:]

    # Save splits
    train_file = output_path / "train.txt"
    val_file = output_path / "val.txt"

    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print(f"Created train ({len(train_lines):,} lines) and val ({len(val_lines):,} lines)")

    return {
        'train': str(train_file),
        'val': str(val_file)
    }


def estimate_dataset_stats(data_path: str, tokenizer) -> dict:
    """
    Estimate dataset statistics.

    Args:
        data_path: Path to text file
        tokenizer: Tokenizer instance

    Returns:
        Dict with statistics
    """
    total_chars = 0
    total_lines = 0
    total_tokens = 0
    sample_tokens = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_chars += len(line)
            total_lines += 1

            if i < 1000:  # Sample first 1000 lines for token stats
                tokens = tokenizer.encode(line)
                total_tokens += len(tokens)
                sample_tokens.extend(tokens)

    # Estimate total tokens
    if total_lines > 0:
        avg_tokens_per_line = total_tokens / min(total_lines, 1000)
        estimated_total_tokens = int(avg_tokens_per_line * total_lines)
    else:
        estimated_total_tokens = 0

    return {
        'total_lines': total_lines,
        'total_chars': total_chars,
        'sampled_tokens': total_tokens,
        'estimated_total_tokens': estimated_total_tokens,
        'avg_tokens_per_line': total_tokens / min(total_lines, 1000) if total_lines > 0 else 0,
        'avg_chars_per_token': total_chars / total_tokens if total_tokens > 0 else 0
    }


if __name__ == "__main__":
    print("Preprocessing module loaded successfully")
