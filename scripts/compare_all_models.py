"""Compare generation quality across all fine-tuned models.

Runs identical prompts through every model and saves a formatted comparison.

Usage:
    python scripts/compare_all_models.py
"""
import sys
import time
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import TinyLlama
from src.tokenizer import Tokenizer


# Test prompts grouped by category
PROMPTS = {
    "Identity": [
        "What is your name?",
        "Who are you?",
    ],
    "Logic & Reasoning": [
        "Which is bigger: an elephant or a mouse?",
        "If it rains, the ground gets wet. The ground is dry. Did it rain?",
        "True or false: All dogs are animals.",
        "Is 10 greater than 7?",
    ],
    "Knowledge": [
        "What is the sun?",
        "Why is the sky blue?",
        "How do plants grow?",
    ],
    "Creative": [
        "Tell me a short story about a brave cat.",
        "Write a short poem about the moon.",
    ],
    "Math (expected to fail)": [
        "What is 2 + 2?",
        "Count from 1 to 5.",
    ],
    "Instruction Following": [
        "List three colors.",
        "Explain what water is in one sentence.",
    ],
}

MODELS = [
    ("checkpoints/226k_v2/finetune/best.pt", "226K v2", "~226K"),
    ("checkpoints/10m_v2/finetune/best.pt", "10M v2 Baseline", "10.7M"),
    ("checkpoints/10m_mtp/finetune/best.pt", "10M MTP", "10.7M"),
    ("checkpoints/10m_synthetic/finetune/best.pt", "10M Synthetic", "10.7M"),
    ("checkpoints/10m_enhanced/finetune/best.pt", "10M Enhanced (DWA+VRL+Reg)", "10.7M"),
    ("checkpoints/dream/finetune/best.pt", "Dream 10M (Ternary+Loop)", "~15M"),
    ("checkpoints/30m_v2/finetune/best.pt", "30M v2 Deep-Narrow", "~30M"),
]


def generate_response(model, tokenizer, prompt, max_tokens=150, temperature=0.7):
    """Generate a response for a single prompt."""
    input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor([input_ids], device="cuda")
    stop_tokens = tokenizer.get_stop_tokens()

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            stop_tokens=stop_tokens,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    generated_ids = output_ids[0][len(input_ids):].tolist()
    response = tokenizer.decode(generated_ids).strip()
    for tag in ["<|im_end|>", "<|im_start|>"]:
        if tag in response:
            response = response.split(tag)[0].strip()
    return response


def main():
    tokenizer = Tokenizer("data/tokenizer_8k.model")
    output_path = Path("exports/model_comparison.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}  # {model_label: {prompt: response}}

    for checkpoint, label, param_count in MODELS:
        if not Path(checkpoint).exists():
            print(f"SKIP: {label} ({checkpoint} not found)")
            continue

        print(f"\nLoading {label}...")
        model = TinyLlama.from_checkpoint(checkpoint, device="cuda")
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,}")

        results = {}
        for category, prompts in PROMPTS.items():
            for prompt in prompts:
                print(f"  [{category}] {prompt[:50]}...", end=" ", flush=True)
                try:
                    response = generate_response(model, tokenizer, prompt)
                    print(f"OK ({len(response)} chars)")
                except Exception as e:
                    response = f"[ERROR: {e}]"
                    print(f"FAIL")
                results[prompt] = response

        all_results[label] = results
        del model
        torch.cuda.empty_cache()

    # Build comparison document
    print(f"\nWriting comparison to {output_path}...")

    lines = []
    lines.append("# Brandon Tiny - Model Comparison Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Models Tested")
    lines.append("")
    lines.append("| Model | Params | Checkpoint |")
    lines.append("|-------|--------|------------|")
    for checkpoint, label, param_count in MODELS:
        exists = "Yes" if Path(checkpoint).exists() else "No"
        lines.append(f"| {label} | {param_count} | {exists} |")
    lines.append("")

    # Per-category comparison
    model_labels = [label for _, label, _ in MODELS if label in all_results]

    for category, prompts in PROMPTS.items():
        lines.append(f"## {category}")
        lines.append("")

        for prompt in prompts:
            lines.append(f"### Prompt: \"{prompt}\"")
            lines.append("")

            for label in model_labels:
                response = all_results[label].get(prompt, "[not tested]")
                # Truncate very long responses
                if len(response) > 400:
                    response = response[:400] + "..."
                lines.append(f"**{label}:**")
                lines.append(f"> {response}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Quality summary
    lines.append("## Quality Summary")
    lines.append("")
    lines.append("| Category | " + " | ".join(model_labels) + " |")
    lines.append("|----------|" + "|".join(["-------"] * len(model_labels)) + "|")

    for category, prompts in PROMPTS.items():
        row = f"| {category} |"
        for label in model_labels:
            # Simple quality heuristic
            scores = []
            for prompt in prompts:
                resp = all_results[label].get(prompt, "")
                # Score based on: not empty, reasonable length, no repetition
                score = 0
                if len(resp) > 10:
                    score += 1
                if 20 < len(resp) < 500:
                    score += 1
                # Check for excessive repetition
                words = resp.split()
                if words and len(set(words)) / max(len(words), 1) > 0.3:
                    score += 1
                scores.append(score)
            avg = sum(scores) / max(len(scores), 1)
            emoji = "Low" if avg < 1.5 else ("Mid" if avg < 2.5 else "Good")
            row += f" {emoji} |"
        lines.append(row)

    lines.append("")

    doc = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc)

    # Also print to stdout
    print("\n" + doc)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
