#!/usr/bin/env python
"""
Interactive chat with TinyLlama.

Usage:
    python scripts/chat.py
    python scripts/chat.py --checkpoint checkpoints/226k/pretrain/best.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.model import TinyLlama
from src.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Chat with TinyLlama")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/226k/finetune/best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/tokenizer.model",
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Max tokens to generate"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = Tokenizer(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = TinyLlama.from_checkpoint(args.checkpoint, device='cpu')
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {model.count_parameters():,} parameters")

    print("\n" + "=" * 60)
    print("🦙 TinyLlama 226K - Interactive Mode")
    print("=" * 60)
    print("Nota: Este modelo solo fue pre-entrenado en wikitext.")
    print("Aún no sabe seguir instrucciones (necesita fine-tuning).")
    print("Escribe texto y el modelo lo continuará.")
    print("Comandos: 'quit' para salir, 'temp X' para cambiar temperatura")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("📝 Tu texto: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 ¡Adiós!")
            break

        if not prompt:
            continue

        if prompt.lower() == 'quit':
            print("👋 ¡Adiós!")
            break

        if prompt.lower().startswith('temp '):
            try:
                args.temperature = float(prompt.split()[1])
                print(f"🌡️  Temperatura: {args.temperature}")
            except:
                print("❌ Uso: temp 0.8")
            continue

        if prompt.lower().startswith('topk '):
            try:
                args.top_k = int(prompt.split()[1])
                print(f"🎯 Top-k: {args.top_k}")
            except:
                print("❌ Uso: topk 50")
            continue

        # Encode prompt
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)

        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )

        # Decode
        generated = tokenizer.decode(output[0].tolist())

        # Highlight the new part
        new_text = generated[len(prompt):]

        print(f"\n🦙 Continuación:{new_text}")
        print("-" * 40 + "\n")


def test_generation():
    """Quick test without interactive mode."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load
    tokenizer = Tokenizer("data/tokenizer.model")
    model = TinyLlama.from_checkpoint("checkpoints/226k/pretrain/best.pt", device='cpu')
    model = model.to(device)
    model.eval()

    # Test prompts
    prompts = [
        "The",
        "In the year",
        "The president of",
        "Machine learning is",
        "The quick brown fox",
    ]

    print("\n" + "=" * 60)
    print("🧪 Test de Generación")
    print("=" * 60)

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50
            )

        generated = tokenizer.decode(output[0].tolist())
        print(f"\n📝 '{prompt}'")
        print(f"🦙 {generated}")
        print("-" * 40)


if __name__ == "__main__":
    # Check if running test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_generation()
    else:
        main()
