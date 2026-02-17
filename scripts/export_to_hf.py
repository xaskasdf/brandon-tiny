"""
Export Brandon-Tiny checkpoint to HuggingFace format.

Converts our PyTorch .pt checkpoint to:
  - model.safetensors (weights in safetensors format)
  - config.json (model architecture config)
  - tokenizer files (copied)
  - README.md (model card, copied from docs/)

Usage:
    python scripts/export_to_hf.py \
        --checkpoint checkpoints/10m_optimal/phase3_finetune/best.pt \
        --output exports/brandon-tiny-10m-instruct \
        --format bf16
"""

import argparse
import json
import os
import shutil
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import TinyLlama


def export_safetensors(model, output_dir, dtype_str='fp32'):
    """Export model weights to safetensors format."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    # Select dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    print(f"  Converting to {dtype_str} ({dtype})")

    # Get state dict and convert dtype
    state_dict = model.state_dict()
    converted = {}
    for key, tensor in state_dict.items():
        if tensor.is_floating_point():
            converted[key] = tensor.to(dtype)
        else:
            converted[key] = tensor

    # Save
    path = os.path.join(output_dir, 'model.safetensors')
    save_file(converted, path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved: {path} ({size_mb:.1f} MB)")
    return size_mb


def export_config(model, output_dir):
    """Export model config as config.json."""
    config = model.config
    config_dict = {
        "architectures": ["TinyLlama"],
        "model_type": "brandon-tiny",
        "dim": config.dim,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "vocab_size": config.vocab_size,
        "hidden_dim": config.hidden_dim,
        "max_seq_len": config.max_seq_len,
        "dropout": config.dropout,
        "weight_tying": config.weight_tying,
        "norm_eps": config.norm_eps,
        "rope_theta": config.rope_theta,
        "block_sharing": config.block_sharing,
        "n_predict": config.n_predict,
        "dense_former": config.dense_former,
        "value_residual": config.value_residual,
        "n_registers": config.n_registers,
        "n_loops": getattr(config, 'n_loops', 1),
        "ternary": getattr(config, 'ternary', False),
        "activation": "swiglu",
        "normalization": "rmsnorm",
        "position_encoding": "rope",
        "chat_format": "chatml",
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }

    path = os.path.join(output_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Saved: {path}")


def export_tokenizer(output_dir):
    """Copy tokenizer files."""
    tokenizer_src = os.path.join('data', 'tokenizer_8k.model')
    vocab_src = os.path.join('data', 'tokenizer_8k.vocab')

    if os.path.exists(tokenizer_src):
        shutil.copy2(tokenizer_src, os.path.join(output_dir, 'tokenizer.model'))
        print(f"  Copied: tokenizer.model")

    if os.path.exists(vocab_src):
        shutil.copy2(vocab_src, os.path.join(output_dir, 'tokenizer.vocab'))
        print(f"  Copied: tokenizer.vocab")

    # Create tokenizer_config.json
    tokenizer_config = {
        "model_type": "sentencepiece",
        "vocab_size": 8192,
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "chat_template": "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n",
        "special_tokens": {
            "im_start": "<|im_start|>",
            "im_end": "<|im_end|>",
            "bos": "<|bos|>",
            "eos": "<|eos|>",
            "pad": "<|pad|>"
        }
    }
    path = os.path.join(output_dir, 'tokenizer_config.json')
    with open(path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"  Saved: {path}")


def export_model_card(output_dir):
    """Copy model card as README.md."""
    model_card_src = os.path.join('docs', 'MODEL_CARD.md')
    if os.path.exists(model_card_src):
        shutil.copy2(model_card_src, os.path.join(output_dir, 'README.md'))
        print(f"  Copied: README.md (from MODEL_CARD.md)")


def main():
    parser = argparse.ArgumentParser(description='Export Brandon-Tiny to HuggingFace format')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/10m_optimal/phase3_finetune/best.pt',
                        help='Path to checkpoint .pt file')
    parser.add_argument('--output', type=str,
                        default='exports/brandon-tiny-10m-instruct',
                        help='Output directory for HuggingFace format')
    parser.add_argument('--format', type=str, default='bf16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Weight format (default: bf16)')
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Brandon-Tiny -> HuggingFace Export")
    print(f"=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.output}")
    print(f"Format:     {args.format}")
    print()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load model
    print("[1/4] Loading checkpoint...")
    model = TinyLlama.from_checkpoint(args.checkpoint, device='cpu')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} parameters")
    print(f"  Config: dim={model.config.dim}, layers={model.config.n_layers}, "
          f"heads={model.config.n_heads}, vocab={model.config.vocab_size}")
    print()

    # Export weights
    print("[2/4] Exporting weights (safetensors)...")
    size_mb = export_safetensors(model, args.output, args.format)
    print()

    # Export config
    print("[3/4] Exporting config...")
    export_config(model, args.output)
    print()

    # Export tokenizer + model card
    print("[4/4] Exporting tokenizer and model card...")
    export_tokenizer(args.output)
    export_model_card(args.output)
    print()

    # Summary
    print(f"=" * 60)
    print(f"Export complete!")
    print(f"  Output directory: {args.output}")
    print(f"  Model size: {size_mb:.1f} MB ({args.format})")
    print(f"  Files:")
    for f in sorted(os.listdir(args.output)):
        fsize = os.path.getsize(os.path.join(args.output, f)) / 1024
        print(f"    {f:30s} {fsize:>8.1f} KB")
    print()
    print(f"To upload to HuggingFace:")
    print(f"  pip install huggingface_hub")
    print(f"  huggingface-cli login")
    print(f"  huggingface-cli upload naranjositos/brandon-tiny-10m-instruct {args.output}")
    print(f"=" * 60)


if __name__ == '__main__':
    main()
