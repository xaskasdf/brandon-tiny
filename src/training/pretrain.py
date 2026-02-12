"""
Pre-training script for TinyLlama.

Usage:
    python src/training/pretrain.py --config configs/model_226k.yaml
    python src/training/pretrain.py --config configs/model_110m.yaml
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model import TinyLlama, ModelConfig
from src.tokenizer import Tokenizer, train_tokenizer
from src.data.dataset import WikitextDataset, TextFileDataset, PretrainDataset, create_dataloader
from src.training.trainer import Trainer, TrainingConfig


def prepare_tokenizer(config: dict, data_path: str) -> Tokenizer:
    """Prepare tokenizer, training if needed."""
    tok_config = config.get('tokenizer', {})
    model_path = tok_config.get('model_path', 'data/tokenizer.model')

    if not Path(model_path).exists():
        print(f"Tokenizer not found at {model_path}, training new tokenizer...")

        # Find training text
        train_text = Path(data_path) / "train.txt"
        if not train_text.exists():
            # Try to prepare wikitext
            from src.data.preprocess import prepare_wikitext
            prepare_wikitext(data_path, version="wikitext-2-raw-v1")

        # Train tokenizer
        vocab_size = tok_config.get('vocab_size', 1024)
        output_prefix = str(Path(model_path).with_suffix(''))
        train_tokenizer(
            input_files=str(train_text),
            output_prefix=output_prefix,
            vocab_size=vocab_size,
            add_chat_tokens=True
        )

    return Tokenizer(model_path)


def main():
    parser = argparse.ArgumentParser(description="Pre-train TinyLlama")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_226k.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (restores optimizer/step)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to initialize weights from (fresh optimizer/step)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory (e.g. data/tinystories, data/fineweb)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        choices=["wikitext", "textfile"],
        help="Override dataset type"
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

    # Check CUDA
    if args.device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Setup paths
    pretrain_config = config.get('pretrain', {})
    data_path = args.data_dir or pretrain_config.get('data_path', 'data/wikitext')
    output_dir = config.get('output_dir', 'checkpoints')

    # Override dataset type if specified
    if args.dataset_type:
        pretrain_config['dataset'] = args.dataset_type
    elif args.data_dir and 'wikitext' not in args.data_dir:
        pretrain_config['dataset'] = 'textfile'

    # Prepare tokenizer
    tokenizer = prepare_tokenizer(config, data_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Update vocab size in model config
    model_config_dict = config['model'].copy()
    if model_config_dict.get('vocab_size') != tokenizer.vocab_size:
        print(f"Warning: config vocab_size ({model_config_dict.get('vocab_size')}) "
              f"!= tokenizer vocab_size ({tokenizer.vocab_size})")
        # Use tokenizer's vocab size
        model_config_dict['vocab_size'] = tokenizer.vocab_size

    # Create model
    model_config = ModelConfig(**model_config_dict)
    model = TinyLlama(model_config)
    print(f"Model parameters: {model.count_parameters():,}")

    # Prepare datasets
    print("Preparing datasets...")
    dataset_type = pretrain_config.get('dataset', 'wikitext')

    if dataset_type == 'wikitext':
        train_dataset = WikitextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            seq_len=model_config.max_seq_len,
            split="train"
        )
        val_dataset = WikitextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            seq_len=model_config.max_seq_len,
            split="validation"
        )
    else:
        # Generic text file dataset (tinystories, fineweb, etc)
        train_dataset = TextFileDataset(
            data_dir=data_path,
            tokenizer=tokenizer,
            seq_len=model_config.max_seq_len,
            split="train"
        )
        val_dataset = TextFileDataset(
            data_dir=data_path,
            tokenizer=tokenizer,
            seq_len=model_config.max_seq_len,
            split="validation"
        )

    # Create data loaders
    batch_size = pretrain_config.get('batch_size', 64)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for Windows compatibility
        drop_last=True
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create training config
    train_config = TrainingConfig(
        learning_rate=pretrain_config.get('learning_rate', 1e-3),
        min_lr=pretrain_config.get('min_lr', 1e-4),
        warmup_iters=pretrain_config.get('warmup_iters', 500),
        max_iters=pretrain_config.get('max_iters', 10000),
        weight_decay=pretrain_config.get('weight_decay', 0.1),
        beta1=pretrain_config.get('beta1', 0.9),
        beta2=pretrain_config.get('beta2', 0.95),
        grad_clip=pretrain_config.get('grad_clip', 1.0),
        lr_schedule=pretrain_config.get('lr_schedule', 'cosine'),
        stable_frac=pretrain_config.get('stable_frac', 0.7),
        decay_frac=pretrain_config.get('decay_frac', 0.2),
        batch_size=batch_size,
        gradient_accumulation_steps=pretrain_config.get('gradient_accumulation_steps', 1),
        log_interval=pretrain_config.get('log_interval', 10),
        eval_interval=pretrain_config.get('eval_interval', 500),
        eval_iters=pretrain_config.get('eval_iters', 100),
        save_interval=pretrain_config.get('save_interval', 1000),
        mtp_curriculum=pretrain_config.get('mtp_curriculum', False),
        mtp_warmup_frac=pretrain_config.get('mtp_warmup_frac', 0.4),
        dtype=pretrain_config.get('dtype', 'bfloat16'),
        compile=pretrain_config.get('compile', False),
        output_dir=os.path.join(output_dir, 'pretrain')
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=args.device
    )

    # Load weights from checkpoint (fresh training state)
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint} (fresh optimizer/step)...")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model weights from checkpoint")

    # Train
    print("\n" + "=" * 50)
    print("Starting pre-training...")
    print("=" * 50 + "\n")

    trainer.train(resume_from=args.resume)

    # Final evaluation
    print("\nFinal evaluation...")
    final_loss = trainer.evaluate()
    print(f"Final validation loss: {final_loss:.4f}")
    print(f"Final perplexity: {math.exp(final_loss):.2f}")


if __name__ == "__main__":
    main()
