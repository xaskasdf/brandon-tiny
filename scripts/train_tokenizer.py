#!/usr/bin/env python
"""
Train a BPE tokenizer for TinyLlama.

Usage:
    python scripts/train_tokenizer.py --vocab_size 1024 --output data/tokenizer_1k
    python scripts/train_tokenizer.py --vocab_size 32000 --output data/tokenizer_32k
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tokenizer import train_tokenizer, Tokenizer, SPECIAL_TOKENS
from src.data.preprocess import prepare_wikitext


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1024,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input text file(s), comma-separated"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tokenizer",
        help="Output prefix (will create .model and .vocab)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram"],
        help="Tokenizer model type"
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=0.9995,
        help="Character coverage ratio"
    )
    parser.add_argument(
        "--prepare_wikitext",
        action="store_true",
        help="Download and prepare wikitext-2 as training data"
    )
    args = parser.parse_args()

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare input data
    if args.input:
        input_files = args.input.split(',')
    elif args.prepare_wikitext or not args.input:
        print("Preparing wikitext-2 as training corpus...")
        data_dir = "data/wikitext"
        paths = prepare_wikitext(data_dir, version="wikitext-2-raw-v1")
        input_files = [paths['train']]
        print(f"Using {input_files[0]} as training corpus")
    else:
        raise ValueError("No input files specified. Use --input or --prepare_wikitext")

    # Verify input files exist
    for f in input_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Input file not found: {f}")

    print(f"\nTraining tokenizer:")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Model type: {args.model_type}")
    print(f"  Input files: {input_files}")
    print(f"  Output: {args.output}")
    print()

    # Train tokenizer
    model_path = train_tokenizer(
        input_files=input_files,
        output_prefix=args.output,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        add_chat_tokens=True
    )

    # Verify tokenizer
    print("\nVerifying tokenizer...")
    tokenizer = Tokenizer(model_path)

    print(f"Actual vocab size: {tokenizer.vocab_size}")

    # Test encoding
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "<|im_start|>user\nWhat is AI?<|im_end|>",
        "<|im_start|>assistant\nAI stands for artificial intelligence.<|im_end|>",
    ]

    print("\nTest encodings:")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"  '{text[:50]}...' -> {len(tokens)} tokens")

    # Check special tokens
    print("\nSpecial tokens:")
    for name, token in SPECIAL_TOKENS.items():
        token_id = tokenizer.piece_to_id(token)
        if token_id != tokenizer.unk_id:
            print(f"  {name}: '{token}' -> {token_id}")
        else:
            print(f"  {name}: '{token}' -> NOT FOUND")

    print(f"\nTokenizer saved to: {model_path}")
    print("Done!")


if __name__ == "__main__":
    main()
