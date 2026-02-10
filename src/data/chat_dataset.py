"""
Chat/Instruction dataset for TinyLlama fine-tuning.

Implements:
- ChatML format parsing
- Target masking (loss only on assistant tokens)
- Support for various instruction datasets
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


class ChatDataset(Dataset):
    """
    Dataset for instruction/chat fine-tuning with target masking.

    Expects data in ChatML format:
    [
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        },
        ...
    ]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int,
        mask_targets_only: bool = True,
        pad_to_max: bool = False
    ):
        """
        Initialize chat dataset.

        Args:
            data_path: Path to JSONL or JSON file with chat data
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            mask_targets_only: If True, only compute loss on assistant tokens
            pad_to_max: If True, pad all sequences to max_seq_len
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_targets_only = mask_targets_only
        self.pad_to_max = pad_to_max

        # Load data
        self.examples = self._load_data(data_path)
        print(f"Loaded {len(self.examples)} chat examples")

    def _load_data(self, data_path: str) -> List[dict]:
        """Load data from JSON or JSONL file."""
        path = Path(data_path)
        examples = []

        if path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                else:
                    examples = [data]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """Get a training example."""
        example = self.examples[idx]
        messages = example.get('messages', example.get('conversation', []))

        # Encode with chat format
        tokens, target_mask = self.tokenizer.encode_chat(messages)

        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            target_mask = target_mask[:self.max_seq_len]

        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

        if self.mask_targets_only:
            mask = torch.tensor(target_mask, dtype=torch.float)
        else:
            mask = torch.ones(len(tokens), dtype=torch.float)

        # Padding
        if self.pad_to_max:
            pad_len = self.max_seq_len - len(tokens)
            if pad_len > 0:
                pad_id = self.tokenizer.pad_id or 0
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), pad_id, dtype=torch.long)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=torch.long)  # Ignore in loss
                ])
                mask = torch.cat([
                    mask,
                    torch.zeros(pad_len, dtype=torch.float)
                ])

        return {
            'input_ids': input_ids,
            'labels': labels,
            'target_mask': mask
        }


class InstructionDataset(Dataset):
    """
    Dataset for instruction-following format.

    Expects data in format:
    {
        "instruction": "...",
        "input": "...",  # optional
        "output": "..."
    }

    Converts to ChatML format internally.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int,
        system_prompt: str = "You are a helpful assistant.",
        mask_targets_only: bool = True
    ):
        """
        Initialize instruction dataset.

        Args:
            data_path: Path to data file
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            system_prompt: System prompt to prepend
            mask_targets_only: Only compute loss on output tokens
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt
        self.mask_targets_only = mask_targets_only

        # Load and convert to chat format
        self.examples = self._load_and_convert(data_path)
        print(f"Loaded {len(self.examples)} instruction examples")

    def _load_and_convert(self, data_path: str) -> List[List[dict]]:
        """Load data and convert to ChatML messages format."""
        path = Path(data_path)
        examples = []

        # Load raw data
        raw_data = []
        if path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

        # Convert to chat format
        for item in raw_data:
            messages = []

            # Add system prompt
            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })

            # Build user message
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')

            if input_text:
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction

            messages.append({
                "role": "user",
                "content": user_content
            })

            # Add assistant response
            output = item.get('output', item.get('response', ''))
            messages.append({
                "role": "assistant",
                "content": output
            })

            examples.append(messages)

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """Get a training example."""
        messages = self.examples[idx]

        # Encode with chat format
        tokens, target_mask = self.tokenizer.encode_chat(messages)

        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            target_mask = target_mask[:self.max_seq_len]

        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

        if self.mask_targets_only:
            mask = torch.tensor(target_mask, dtype=torch.float)
        else:
            mask = torch.ones(len(tokens), dtype=torch.float)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'target_mask': mask
        }


def collate_chat(batch: List[dict], pad_id: int = 0) -> dict:
    """
    Collate function for chat batches with variable lengths.

    Pads to the longest sequence in the batch.
    """
    # Find max length in batch
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids = []
    labels = []
    target_masks = []

    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len

        # Pad input_ids
        input_ids.append(torch.cat([
            item['input_ids'],
            torch.full((pad_len,), pad_id, dtype=torch.long)
        ]))

        # Pad labels with -100 (ignored by cross_entropy)
        labels.append(torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ]))

        # Pad mask with 0
        target_masks.append(torch.cat([
            item['target_mask'],
            torch.zeros(pad_len, dtype=torch.float)
        ]))

    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'target_mask': torch.stack(target_masks)
    }


def create_synthetic_data(
    output_path: str,
    num_examples: int = 1000,
    seed: int = 42
):
    """
    Create synthetic instruction data for testing.

    Generates simple Q&A pairs for basic model testing.
    """
    random.seed(seed)

    templates = [
        # Math
        {
            "instruction": "What is {a} + {b}?",
            "output": "{a} + {b} = {c}",
            "gen": lambda: {"a": random.randint(1, 100), "b": random.randint(1, 100)},
            "compute": lambda d: {"c": d["a"] + d["b"]}
        },
        {
            "instruction": "What is {a} - {b}?",
            "output": "{a} - {b} = {c}",
            "gen": lambda: {"a": random.randint(50, 100), "b": random.randint(1, 50)},
            "compute": lambda d: {"c": d["a"] - d["b"]}
        },
        {
            "instruction": "What is {a} * {b}?",
            "output": "{a} * {b} = {c}",
            "gen": lambda: {"a": random.randint(1, 12), "b": random.randint(1, 12)},
            "compute": lambda d: {"c": d["a"] * d["b"]}
        },
        # Simple facts
        {
            "instruction": "What color is the sky?",
            "output": "The sky is blue.",
            "gen": lambda: {},
            "compute": lambda d: {}
        },
        {
            "instruction": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "gen": lambda: {},
            "compute": lambda d: {}
        },
        {
            "instruction": "How many days are in a week?",
            "output": "There are 7 days in a week.",
            "gen": lambda: {},
            "compute": lambda d: {}
        },
        # Conversions
        {
            "instruction": "Convert {n} meters to centimeters.",
            "output": "{n} meters = {cm} centimeters.",
            "gen": lambda: {"n": random.randint(1, 10)},
            "compute": lambda d: {"cm": d["n"] * 100}
        },
        # Simple reasoning
        {
            "instruction": "Is {n} even or odd?",
            "output": "{n} is {parity}.",
            "gen": lambda: {"n": random.randint(1, 100)},
            "compute": lambda d: {"parity": "even" if d["n"] % 2 == 0 else "odd"}
        },
    ]

    examples = []
    for _ in range(num_examples):
        template = random.choice(templates)
        data = template["gen"]()
        data.update(template["compute"](data))

        examples.append({
            "instruction": template["instruction"].format(**data),
            "output": template["output"].format(**data)
        })

    # Save as JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Created {num_examples} synthetic examples at {output_path}")


if __name__ == "__main__":
    # Create sample data for testing
    create_synthetic_data("data/synthetic_instructions.jsonl", num_examples=100)
