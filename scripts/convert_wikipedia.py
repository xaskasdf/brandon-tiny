#!/usr/bin/env python
"""
Convert Wikipedia parquet dump to training format.

Usage:
    python scripts/convert_wikipedia.py --input data/wikipedia_parquet --output data/wikipedia
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import (
    process_wikipedia_parquet,
    shard_tokenized_data,
    create_train_val_split,
    estimate_dataset_stats
)
from src.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Convert Wikipedia parquet to training format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing parquet files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wikipedia",
        help="Output directory"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/tokenizer_32k.model",
        help="Path to tokenizer model"
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=None,
        help="Maximum number of articles to process"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100_000_000,
        help="Number of tokens per shard"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.01,
        help="Validation split ratio"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert parquet to text
    print("=" * 50)
    print("Step 1: Converting parquet to text")
    print("=" * 50)

    text_file = output_path / "wikipedia.txt"
    if not text_file.exists():
        num_articles = process_wikipedia_parquet(
            input_dir=args.input,
            output_file=str(text_file),
            max_articles=args.max_articles
        )
        print(f"Processed {num_articles} articles")
    else:
        print(f"Text file already exists: {text_file}")

    # Step 2: Create train/val split
    print("\n" + "=" * 50)
    print("Step 2: Creating train/val split")
    print("=" * 50)

    split_paths = create_train_val_split(
        input_file=str(text_file),
        output_dir=str(output_path),
        val_ratio=args.val_ratio
    )

    # Step 3: Load tokenizer
    print("\n" + "=" * 50)
    print("Step 3: Loading tokenizer")
    print("=" * 50)

    if not Path(args.tokenizer).exists():
        print(f"Tokenizer not found: {args.tokenizer}")
        print("Please train a tokenizer first using scripts/train_tokenizer.py")
        sys.exit(1)

    tokenizer = Tokenizer(args.tokenizer)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Step 4: Estimate statistics
    print("\n" + "=" * 50)
    print("Step 4: Estimating dataset statistics")
    print("=" * 50)

    stats = estimate_dataset_stats(split_paths['train'], tokenizer)
    print(f"Train set statistics:")
    print(f"  Total lines: {stats['total_lines']:,}")
    print(f"  Total chars: {stats['total_chars']:,}")
    print(f"  Estimated tokens: {stats['estimated_total_tokens']:,}")
    print(f"  Avg tokens/line: {stats['avg_tokens_per_line']:.1f}")
    print(f"  Avg chars/token: {stats['avg_chars_per_token']:.1f}")

    # Step 5: Tokenize and shard
    print("\n" + "=" * 50)
    print("Step 5: Tokenizing and sharding")
    print("=" * 50)

    for split in ['train', 'val']:
        print(f"\nProcessing {split} split...")
        total_tokens = shard_tokenized_data(
            input_file=split_paths[split],
            output_dir=str(output_path / "tokenized"),
            tokenizer=tokenizer,
            shard_size=args.shard_size,
            split=split
        )
        print(f"Total {split} tokens: {total_tokens:,}")

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)
    print(f"\nOutput directory: {output_path}")
    print(f"Tokenized data: {output_path / 'tokenized'}")


if __name__ == "__main__":
    main()
