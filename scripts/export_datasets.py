"""
Export Brandon-Tiny datasets to HuggingFace format (parquet).

Exports:
  1. Instruction dataset (chat/train.jsonl + val.jsonl) -> parquet
  2. Synthetic pretrain sample -> parquet

Usage:
    python scripts/export_datasets.py --output exports/datasets
"""

import argparse
import json
import os
import sys


def jsonl_to_records(path):
    """Read JSONL file and return list of records."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def export_instruction_dataset(output_dir):
    """Export instruction dataset (chat JSONL) to parquet."""
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Run: pip install pandas pyarrow")
        sys.exit(1)

    train_path = os.path.join('data', 'chat', 'train.jsonl')
    val_path = os.path.join('data', 'chat', 'val.jsonl')

    if not os.path.exists(train_path):
        print(f"  SKIP: {train_path} not found")
        return

    inst_dir = os.path.join(output_dir, 'brandon-tiny-instruct')
    os.makedirs(inst_dir, exist_ok=True)

    # Train split
    print(f"  Reading {train_path}...")
    train_records = jsonl_to_records(train_path)
    print(f"  {len(train_records):,} training examples")

    # Normalize records - extract system/user/assistant from messages
    train_flat = []
    for rec in train_records:
        flat = {}
        messages = rec.get('messages', [])
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                flat['system'] = content
            elif role == 'user':
                flat['instruction'] = content
            elif role == 'assistant':
                flat['response'] = content
        # Fallback for non-messages format
        if 'instruction' not in flat:
            flat['instruction'] = rec.get('instruction', rec.get('prompt', ''))
            flat['response'] = rec.get('response', rec.get('output', ''))
            flat['system'] = rec.get('system', '')
        # Keep original messages too
        flat['messages'] = json.dumps(messages) if messages else ''
        train_flat.append(flat)

    df_train = pd.DataFrame(train_flat)
    train_parquet = os.path.join(inst_dir, 'train.parquet')
    df_train.to_parquet(train_parquet, index=False)
    size_mb = os.path.getsize(train_parquet) / (1024 * 1024)
    print(f"  Saved: {train_parquet} ({size_mb:.1f} MB, {len(df_train):,} rows)")

    # Val split
    if os.path.exists(val_path):
        print(f"  Reading {val_path}...")
        val_records = jsonl_to_records(val_path)

        val_flat = []
        for rec in val_records:
            flat = {}
            messages = rec.get('messages', [])
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'system':
                    flat['system'] = content
                elif role == 'user':
                    flat['instruction'] = content
                elif role == 'assistant':
                    flat['response'] = content
            if 'instruction' not in flat:
                flat['instruction'] = rec.get('instruction', rec.get('prompt', ''))
                flat['response'] = rec.get('response', rec.get('output', ''))
                flat['system'] = rec.get('system', '')
            flat['messages'] = json.dumps(messages) if messages else ''
            val_flat.append(flat)

        df_val = pd.DataFrame(val_flat)
        val_parquet = os.path.join(inst_dir, 'validation.parquet')
        df_val.to_parquet(val_parquet, index=False)
        size_mb = os.path.getsize(val_parquet) / (1024 * 1024)
        print(f"  Saved: {val_parquet} ({size_mb:.1f} MB, {len(df_val):,} rows)")

    # Dataset card
    card = """---
license: apache-2.0
language:
  - en
task_categories:
  - text-generation
  - conversational
tags:
  - instruction-following
  - chatml
  - synthetic
  - curated
size_categories:
  - 10K<n<100K
---

# Brandon-Tiny Instruct Dataset

Instruction fine-tuning dataset used to train [Brandon-Tiny-10M-Instruct](https://huggingface.co/naranjositos/brandon-tiny-10m-instruct).

## Dataset Details

- **Train:** {train_n:,} examples
- **Validation:** {val_n:,} examples
- **Format:** ChatML (system/user/assistant messages)
- **Language:** English

## Composition

- ~57,000 curated chat instructions (general knowledge, creative writing, explanations)
- ~19,944 reasoning/Chain-of-Thought examples (math, logic, science)
- ~200 pretrain replay examples (text continuation, catastrophic forgetting mitigation)

## Format

Each example has:
- `instruction`: User query/prompt
- `response`: Assistant response
- `system`: System prompt (if any)
- `messages`: Full ChatML message list as JSON string

## Usage

```python
from datasets import load_dataset
ds = load_dataset("naranjositos/brandon-tiny-instruct")
```

## Citation

```bibtex
@misc{{brandon-tiny-2026,
  title={{Brandon-Tiny 10M: A 3-Phase Training Pipeline for Ultra-Small Instruction-Following Language Models}},
  author={{Samuel Cortes}},
  year={{2026}},
  url={{https://naranjositos.tech/}}
}}
```
"""
    val_n = len(val_records) if os.path.exists(val_path) else 0
    card = card.format(train_n=len(train_records), val_n=val_n)
    card_path = os.path.join(inst_dir, 'README.md')
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(card)
    print(f"  Saved: {card_path}")


def export_synthetic_pretrain(output_dir, max_examples=50000):
    """Export a sample of synthetic pretrain data as parquet."""
    try:
        import pandas as pd
    except ImportError:
        return

    txt_path = os.path.join('data', 'synthetic_pretrain', 'train_all.txt')
    if not os.path.exists(txt_path):
        print(f"  SKIP: {txt_path} not found")
        return

    synth_dir = os.path.join(output_dir, 'brandon-tiny-synthetic-pretrain')
    os.makedirs(synth_dir, exist_ok=True)

    print(f"  Reading {txt_path} (sampling {max_examples:,} passages)...")

    # Read and split into passages (separated by double newlines)
    records = []
    current_passage = []
    count = 0
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '' and current_passage:
                text = '\n'.join(current_passage).strip()
                if len(text) > 50:  # Skip very short passages
                    records.append({'text': text})
                    count += 1
                    if count >= max_examples:
                        break
                current_passage = []
            else:
                current_passage.append(line.rstrip())

    # Add last passage
    if current_passage and count < max_examples:
        text = '\n'.join(current_passage).strip()
        if len(text) > 50:
            records.append({'text': text})

    print(f"  Extracted {len(records):,} passages")

    df = pd.DataFrame(records)
    parquet_path = os.path.join(synth_dir, 'train.parquet')
    df.to_parquet(parquet_path, index=False)
    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"  Saved: {parquet_path} ({size_mb:.1f} MB, {len(df):,} rows)")

    # Dataset card
    card = """---
license: apache-2.0
language:
  - en
task_categories:
  - text-generation
tags:
  - synthetic
  - pretrain
  - gpt-4o-mini
size_categories:
  - 10K<n<100K
---

# Brandon-Tiny Synthetic Pretrain Data

Synthetic pre-training data generated by GPT-4o-mini, used as 30% of the pre-training mixture for [Brandon-Tiny-10M-Instruct](https://huggingface.co/naranjositos/brandon-tiny-10m-instruct).

## Details

- **Source:** Generated via GPT-4o-mini API from diverse topic seeds
- **Format:** Plain text passages
- **Language:** English
- **Purpose:** Diverse topic coverage to complement Wikipedia and SmolLM corpus

This is a sample of the full synthetic dataset used in training.

## Citation

```bibtex
@misc{{brandon-tiny-2026,
  title={{Brandon-Tiny 10M: A 3-Phase Training Pipeline for Ultra-Small Instruction-Following Language Models}},
  author={{Samuel Cortes}},
  year={{2026}},
  url={{https://naranjositos.tech/}}
}}
```
"""
    card_path = os.path.join(synth_dir, 'README.md')
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(card)
    print(f"  Saved: {card_path}")


def main():
    parser = argparse.ArgumentParser(description='Export datasets to HuggingFace format')
    parser.add_argument('--output', type=str, default='exports/datasets',
                        help='Output directory')
    parser.add_argument('--max-synthetic', type=int, default=50000,
                        help='Max synthetic pretrain passages to export')
    args = parser.parse_args()

    print("=" * 60)
    print("Brandon-Tiny Dataset Export")
    print("=" * 60)
    print()

    print("[1/2] Exporting instruction dataset...")
    export_instruction_dataset(args.output)
    print()

    print("[2/2] Exporting synthetic pretrain sample...")
    export_synthetic_pretrain(args.output, args.max_synthetic)
    print()

    print("=" * 60)
    print("Export complete!")
    print()
    print("To upload to HuggingFace:")
    print("  pip install huggingface_hub")
    print("  huggingface-cli login")
    print(f"  huggingface-cli upload naranjositos/brandon-tiny-instruct {args.output}/brandon-tiny-instruct --repo-type dataset")
    print(f"  huggingface-cli upload naranjositos/brandon-tiny-synthetic-pretrain {args.output}/brandon-tiny-synthetic-pretrain --repo-type dataset")
    print("=" * 60)


if __name__ == '__main__':
    main()
