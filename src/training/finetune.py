"""
Fine-tuning script for TinyLlama instruction/chat models.

Usage:
    python src/training/finetune.py --config configs/model_226k.yaml --checkpoint checkpoints/pretrain/best.pt
    python src/training/finetune.py --config configs/model_110m.yaml --checkpoint checkpoints/pretrain/best.pt
"""

import argparse
import os
import sys
from pathlib import Path
from functools import partial

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model import TinyLlama, ModelConfig
from src.tokenizer import Tokenizer
from src.data.chat_dataset import (
    ChatDataset,
    InstructionDataset,
    collate_chat,
    create_synthetic_data
)
from src.training.trainer import Trainer, TrainingConfig


def prepare_data(config: dict, tokenizer: Tokenizer) -> tuple:
    """Prepare fine-tuning datasets."""
    finetune_config = config.get('finetune', {})
    data_path = finetune_config.get('data_path', 'data/chat')
    max_seq_len = config['model'].get('max_seq_len', 128)

    data_dir = Path(data_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing data
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    if not train_file.exists():
        print("No training data found, creating synthetic data...")
        create_synthetic_data(str(train_file), num_examples=2000)

    if not val_file.exists():
        print("No validation data found, creating synthetic data...")
        create_synthetic_data(str(val_file), num_examples=200)

    # Create datasets
    train_dataset = InstructionDataset(
        data_path=str(train_file),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        system_prompt="You are a helpful assistant.",
        mask_targets_only=finetune_config.get('mask_targets_only', True)
    )

    val_dataset = InstructionDataset(
        data_path=str(val_file),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        system_prompt="You are a helpful assistant.",
        mask_targets_only=finetune_config.get('mask_targets_only', True)
    )

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama for chat/instruction")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_226k.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained checkpoint"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to fine-tuning checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {args.config}")
    print(f"Device: {args.device}")

    # Load tokenizer
    tok_config = config.get('tokenizer', {})
    model_path = tok_config.get('model_path', 'data/tokenizer.model')

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {model_path}. "
            "Run pre-training first or train tokenizer separately."
        )

    tokenizer = Tokenizer(model_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load or create model
    model_config_dict = config['model'].copy()
    model_config_dict['vocab_size'] = tokenizer.vocab_size

    if args.checkpoint:
        print(f"Loading pre-trained model from {args.checkpoint}")
        model = TinyLlama.from_checkpoint(args.checkpoint, device=args.device)
        # Move to CPU first for loading
        model = model.cpu()
    else:
        print("No checkpoint provided, initializing new model")
        model_config = ModelConfig(**model_config_dict)
        model = TinyLlama(model_config)

    print(f"Model parameters: {model.count_parameters():,}")

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset = prepare_data(config, tokenizer)

    # Create data loaders with custom collate
    finetune_config = config.get('finetune', {})
    batch_size = finetune_config.get('batch_size', 32)

    collate_fn = partial(collate_chat, pad_id=tokenizer.pad_id or 0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Create training config for fine-tuning
    output_dir = config.get('output_dir', 'checkpoints')
    train_config = TrainingConfig(
        learning_rate=finetune_config.get('learning_rate', 2e-5),
        min_lr=finetune_config.get('min_lr', 2e-6),
        warmup_iters=finetune_config.get('warmup_iters', 100),
        max_iters=finetune_config.get('max_iters', 5000),
        weight_decay=finetune_config.get('weight_decay', 0.01),
        beta1=finetune_config.get('beta1', 0.9),
        beta2=finetune_config.get('beta2', 0.95),
        grad_clip=finetune_config.get('grad_clip', 1.0),
        lr_schedule=finetune_config.get('lr_schedule', 'cosine'),
        stable_frac=finetune_config.get('stable_frac', 0.7),
        decay_frac=finetune_config.get('decay_frac', 0.2),
        batch_size=batch_size,
        gradient_accumulation_steps=finetune_config.get('gradient_accumulation_steps', 2),
        log_interval=finetune_config.get('log_interval', 10),
        eval_interval=finetune_config.get('eval_interval', 250),
        eval_iters=finetune_config.get('eval_iters', 50),
        save_interval=finetune_config.get('save_interval', 500),
        dtype=finetune_config.get('dtype', 'bfloat16'),
        compile=finetune_config.get('compile', False),
        output_dir=os.path.join(output_dir, 'finetune')
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=args.device
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting fine-tuning...")
    print("=" * 50 + "\n")

    trainer.train(resume_from=args.resume)

    # Test generation
    print("\nTesting generation...")
    model.eval()
    model = model.to(args.device)

    test_prompt = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n"
    tokens = tokenizer.encode(test_prompt)
    input_ids = torch.tensor([tokens], device=args.device)

    gen_config = config.get('generation', {})
    output = model.generate(
        input_ids,
        max_new_tokens=gen_config.get('max_new_tokens', 64),
        temperature=gen_config.get('temperature', 0.8),
        top_p=gen_config.get('top_p', 0.9),
        top_k=gen_config.get('top_k', 40),
        stop_tokens=tokenizer.get_stop_tokens(),
        repetition_penalty=gen_config.get('repetition_penalty', 1.0),
        no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 0),
    )

    generated_text = tokenizer.decode(output[0].tolist())
    print(f"\nGenerated:\n{generated_text}")

    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()
