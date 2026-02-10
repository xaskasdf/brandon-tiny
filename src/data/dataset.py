"""
Dataset classes for TinyLlama pre-training.

Supports:
- Wikitext dataset (for 226K model)
- Wikipedia dataset (for 110M model)
- Efficient streaming and caching
"""

import os
import pickle
from pathlib import Path
from typing import Optional, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class PretrainDataset(Dataset):
    """
    Pre-training dataset that serves fixed-length chunks of tokenized text.

    This is a map-style dataset that loads pre-tokenized data from disk.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        split: str = "train",
        tokenizer=None
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to directory containing tokenized data
            seq_len: Sequence length (context window)
            split: "train" or "val"
            tokenizer: Optional tokenizer for on-the-fly tokenization
        """
        self.seq_len = seq_len
        self.split = split
        self.tokenizer = tokenizer

        # Load tokenized data as memory-mapped array
        data_file = Path(data_path) / f"{split}.bin"

        if data_file.exists():
            # Load pre-tokenized data
            self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
            print(f"Loaded {split} data: {len(self.data):,} tokens")
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                "Run preprocessing first to create tokenized data."
            )

    def __len__(self) -> int:
        # Number of complete sequences we can form
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> dict:
        """Get a training example."""
        # Get chunk of tokens
        tokens = self.data[idx:idx + self.seq_len + 1].astype(np.int64)

        # Input and target (shifted by 1)
        x = torch.from_numpy(tokens[:-1])
        y = torch.from_numpy(tokens[1:])

        return {'input_ids': x, 'labels': y}


class StreamingPretrainDataset(IterableDataset):
    """
    Streaming dataset for large-scale pre-training.

    Efficiently streams tokenized data without loading everything into memory.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        split: str = "train",
        shuffle_shards: bool = True,
        seed: int = 42
    ):
        """
        Initialize streaming dataset.

        Args:
            data_path: Path to directory containing sharded data files
            seq_len: Sequence length
            split: "train" or "val"
            shuffle_shards: Whether to shuffle shard order
            seed: Random seed for shuffling
        """
        self.seq_len = seq_len
        self.split = split
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        # Find all shard files
        data_dir = Path(data_path)
        self.shards = sorted(data_dir.glob(f"{split}_*.bin"))

        if not self.shards:
            # Fall back to single file
            single_file = data_dir / f"{split}.bin"
            if single_file.exists():
                self.shards = [single_file]
            else:
                raise FileNotFoundError(f"No data files found in {data_path}")

        print(f"Found {len(self.shards)} shards for {split}")

    def __iter__(self) -> Iterator[dict]:
        """Iterate over examples."""
        worker_info = torch.utils.data.get_worker_info()

        # Determine which shards this worker handles
        if worker_info is None:
            shards = self.shards
            worker_id = 0
        else:
            # Split shards among workers
            per_worker = len(self.shards) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.shards)
            shards = self.shards[start:end]

        # Shuffle shards
        if self.shuffle_shards:
            rng = np.random.RandomState(self.seed + worker_id)
            shards = list(shards)
            rng.shuffle(shards)

        # Stream data from shards
        buffer = []
        for shard_path in shards:
            data = np.memmap(shard_path, dtype=np.uint16, mode='r')

            for i in range(0, len(data) - self.seq_len, self.seq_len):
                tokens = data[i:i + self.seq_len + 1].astype(np.int64)
                x = torch.from_numpy(tokens[:-1].copy())
                y = torch.from_numpy(tokens[1:].copy())

                yield {'input_ids': x, 'labels': y}


class WikitextDataset(Dataset):
    """
    Dataset for Wikitext-2 or Wikitext-103.

    Handles downloading and tokenization.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_len: int,
        split: str = "train"
    ):
        """
        Initialize Wikitext dataset.

        Args:
            data_path: Path to cache directory
            tokenizer: Tokenizer instance
            seq_len: Sequence length
            split: "train", "validation", or "test"
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        cache_file = Path(data_path) / f"wikitext_{split}_tokens.pkl"

        if cache_file.exists():
            print(f"Loading cached tokens from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.tokens = pickle.load(f)
        else:
            print(f"Tokenizing wikitext-{split}...")
            self.tokens = self._prepare_data(data_path, split, cache_file)

        print(f"Loaded {len(self.tokens):,} tokens for {split}")

    def _prepare_data(self, data_path: str, split: str, cache_file: Path) -> list:
        """Download and tokenize data."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Tokenize all text
        tokens = []
        for example in dataset:
            text = example['text']
            if text.strip():  # Skip empty lines
                tokens.extend(self.tokenizer.encode(text))

        # Cache tokens
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(tokens, f)

        return tokens

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return {'input_ids': x, 'labels': y}


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True
) -> torch.utils.data.DataLoader:
    """Create a DataLoader with sensible defaults."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def tokenize_and_save(
    input_path: str,
    output_path: str,
    tokenizer,
    split: str = "train",
    shard_size: int = 100_000_000  # 100M tokens per shard
):
    """
    Tokenize a text file and save as binary format.

    Args:
        input_path: Path to input text file
        output_path: Output directory
        tokenizer: Tokenizer instance
        split: Data split name
        shard_size: Number of tokens per shard file
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    shard_idx = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tokens = tokenizer.encode(line)
                all_tokens.extend(tokens)

                # Save shard if buffer is full
                if len(all_tokens) >= shard_size:
                    shard_file = output_dir / f"{split}_{shard_idx:04d}.bin"
                    arr = np.array(all_tokens[:shard_size], dtype=np.uint16)
                    arr.tofile(shard_file)
                    print(f"Saved shard {shard_idx}: {shard_size:,} tokens")

                    all_tokens = all_tokens[shard_size:]
                    shard_idx += 1

    # Save remaining tokens
    if all_tokens:
        if shard_idx == 0:
            # Single file
            shard_file = output_dir / f"{split}.bin"
        else:
            shard_file = output_dir / f"{split}_{shard_idx:04d}.bin"

        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(shard_file)
        print(f"Saved final shard: {len(all_tokens):,} tokens")


if __name__ == "__main__":
    # Quick test
    print("Dataset module loaded successfully")
