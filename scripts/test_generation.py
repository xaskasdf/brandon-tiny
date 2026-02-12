"""Test generation quality of fine-tuned models."""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import TinyLlama
from src.tokenizer import Tokenizer


def test_model(checkpoint_path, tokenizer, prompts, label, max_tokens=128, temperature=0.7, top_k=50):
    """Test a model with multiple prompts."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    model = TinyLlama.from_checkpoint(checkpoint_path, device="cuda")
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    stop_tokens = tokenizer.get_stop_tokens()
    print(f"  Stop tokens: {stop_tokens}")
    print()

    for i, prompt in enumerate(prompts):
        # Format as ChatML
        input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids], device="cuda")

        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=0.9,
                stop_tokens=stop_tokens
            )

        # Decode only the generated part (after input)
        generated_ids = output_ids[0][len(input_ids):].tolist()
        response = tokenizer.decode(generated_ids).strip()
        # Clean up any trailing special tokens
        for tag in ["<|im_end|>", "<|im_start|>"]:
            if tag in response:
                response = response.split(tag)[0].strip()

        print(f"  [{i+1}] User: {prompt}")
        print(f"      Assistant: {response[:300]}")
        print()

    del model
    torch.cuda.empty_cache()


def main():
    tokenizer = Tokenizer("data/tokenizer_8k.model")

    prompts = [
        "What is 2 + 2?",
        "Tell me a short story about a cat.",
        "What is the sun?",
        "How do plants grow?",
        "Say hello in Spanish.",
        "Why is the sky blue?",
        "What is your name?",
        "Count from 1 to 5.",
    ]

    models = [
        ("checkpoints/10m_v2/finetune/best.pt", "10M v2 Baseline (val_loss ~3.92)"),
        ("checkpoints/10m_mtp/finetune/best.pt", "10M MTP n_predict=4 (val_loss 3.90)"),
        ("checkpoints/30m_v2/finetune/best.pt", "30M v2 Deep-Narrow (val_loss 2.61)"),
    ]

    for checkpoint, label in models:
        if Path(checkpoint).exists():
            test_model(checkpoint, tokenizer, prompts, label)
        else:
            print(f"Checkpoint not found: {checkpoint}")

    print("\n" + "="*60)
    print("  Generation test complete!")
    print("="*60)


if __name__ == "__main__":
    main()
