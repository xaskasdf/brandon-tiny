#!/usr/bin/env python
"""
Demo script for TinyLlama.

Quick demonstration of model capabilities.

Usage:
    python scripts/demo.py
    python scripts/demo.py --checkpoint checkpoints/finetune/best.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def demo_model_creation():
    """Demonstrate model creation."""
    from src.model import TinyLlama, ModelConfig

    print("=" * 60)
    print("Demo: Model Creation")
    print("=" * 60)

    # Create 226K model
    config = ModelConfig(
        dim=48,
        n_layers=7,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=1024,
        hidden_dim=128,
        max_seq_len=128,
        dropout=0.15,
        weight_tying=True
    )

    model = TinyLlama(config)

    print(f"\nModel created with {model.count_parameters():,} parameters")
    print(f"\nArchitecture:")
    print(f"  - {config.n_layers} transformer layers")
    print(f"  - {config.n_heads} attention heads")
    print(f"  - {config.n_kv_heads} key-value heads (GQA)")
    print(f"  - {config.dim} embedding dimension")
    print(f"  - {config.hidden_dim} FFN hidden dimension")
    print(f"  - {config.vocab_size} vocabulary size")
    print(f"  - {config.max_seq_len} max sequence length")

    return model


def demo_inference():
    """Demonstrate inference."""
    from src.model import TinyLlama, ModelConfig

    print("\n" + "=" * 60)
    print("Demo: Inference")
    print("=" * 60)

    # Create small model for demo
    config = ModelConfig(
        dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=256,
        hidden_dim=64,
        max_seq_len=64
    )

    model = TinyLlama(config)
    model.eval()

    # Random input
    input_ids = torch.randint(0, 256, (1, 10))

    with torch.no_grad():
        # Forward pass
        logits, _ = model(input_ids)
        print(f"\nForward pass:")
        print(f"  Input: {input_ids.shape}")
        print(f"  Output: {logits.shape}")

        # Generation
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50
        )
        print(f"\nGeneration:")
        print(f"  Prompt: {input_ids[0].tolist()}")
        print(f"  Generated: {generated[0].tolist()}")


def demo_with_tokenizer():
    """Demonstrate with tokenizer if available."""
    from src.model import TinyLlama, ModelConfig

    print("\n" + "=" * 60)
    print("Demo: Text Generation")
    print("=" * 60)

    tokenizer_path = Path("data/tokenizer.model")

    if not tokenizer_path.exists():
        # Use alternative tokenizer path
        alt_paths = [
            Path("data/tokenizer_1k.model"),
            Path("data/test_tokenizer.model")
        ]
        for path in alt_paths:
            if path.exists():
                tokenizer_path = path
                break

    if not tokenizer_path.exists():
        print("\nNo tokenizer found. Skipping text demo.")
        print("Train a tokenizer first: python scripts/train_tokenizer.py")
        return

    from src.tokenizer import Tokenizer

    tokenizer = Tokenizer(str(tokenizer_path))
    print(f"\nLoaded tokenizer with {tokenizer.vocab_size} tokens")

    # Create model matching tokenizer
    config = ModelConfig(
        dim=48,
        n_layers=7,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=tokenizer.vocab_size,
        hidden_dim=128,
        max_seq_len=128,
        dropout=0.0
    )

    model = TinyLlama(config)
    model.eval()

    print(f"Created model with {model.count_parameters():,} parameters")

    # Test prompts
    prompts = [
        "Hello, how are you?",
        "<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n",
    ]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens])

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=30,
                temperature=0.8,
                top_k=40,
                stop_tokens=tokenizer.get_stop_tokens()
            )

        generated_text = tokenizer.decode(output[0].tolist())
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Generated: {generated_text[:100]}...")


def demo_chat_format():
    """Demonstrate chat format."""
    print("\n" + "=" * 60)
    print("Demo: Chat Format (ChatML)")
    print("=" * 60)

    tokenizer_path = Path("data/tokenizer.model")
    for path in [Path("data/tokenizer_1k.model"), Path("data/test_tokenizer.model")]:
        if path.exists():
            tokenizer_path = path
            break

    if not tokenizer_path.exists():
        print("\nNo tokenizer found. Creating example format...")

        example = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>"""

        print(f"\nChatML format:\n{example}")
        return

    from src.tokenizer import Tokenizer

    tokenizer = Tokenizer(str(tokenizer_path))

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 = 4."},
    ]

    tokens, mask = tokenizer.encode_chat(messages)

    print(f"\nMessages: {len(messages)}")
    print(f"Total tokens: {len(tokens)}")
    print(f"Assistant tokens (for loss): {sum(mask)}")
    print(f"\nToken sequence: {tokens[:20]}...")
    print(f"Target mask: {mask[:20]}...")


def demo_from_checkpoint(checkpoint_path: str):
    """Demonstrate loading from checkpoint."""
    from src.model import TinyLlama

    print("\n" + "=" * 60)
    print(f"Demo: Loading Checkpoint")
    print("=" * 60)

    model = TinyLlama.from_checkpoint(checkpoint_path)
    print(f"\nLoaded model with {model.count_parameters():,} parameters")
    print(f"Config: dim={model.config.dim}, layers={model.config.n_layers}")

    return model


def main():
    parser = argparse.ArgumentParser(description="TinyLlama Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TinyLlama Demo")
    print("=" * 60)

    # Basic demos
    demo_model_creation()
    demo_inference()

    # Optional demos
    demo_with_tokenizer()
    demo_chat_format()

    # Checkpoint demo
    if args.checkpoint and Path(args.checkpoint).exists():
        demo_from_checkpoint(args.checkpoint)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
