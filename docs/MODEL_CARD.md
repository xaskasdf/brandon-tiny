---
license: apache-2.0
language:
  - en
library_name: pytorch
pipeline_tag: text-generation
tags:
  - llama
  - text-generation
  - conversational
  - tiny
  - ultra-small
  - instruction-following
  - knowledge-distillation
  - deep-narrow
  - mobilellm
  - denseformer
datasets:
  - wikimedia/wikipedia
  - HuggingFaceTB/smollm-corpus
  - nyu-mll/blimp
  - Rowan/hellaswag
  - allenai/ai2_arc
  - ybisk/piqa
  - allenai/winogrande
  - cimec/lambada
  - Salesforce/wikitext
model-index:
  - name: Brandon-Tiny-10M-Instruct
    results:
      - task:
          type: text-generation
          name: BLiMP (Grammatical Knowledge)
        dataset:
          name: BLiMP
          type: blimp
        metrics:
          - type: accuracy
            value: 73.3
            name: Accuracy
      - task:
          type: text-generation
          name: HellaSwag (Commonsense)
        dataset:
          name: HellaSwag
          type: Rowan/hellaswag
        metrics:
          - type: accuracy
            value: 32.4
            name: 0-shot Accuracy
      - task:
          type: text-generation
          name: ARC-Easy (Science Reasoning)
        dataset:
          name: ARC-Easy
          type: allenai/ai2_arc
        metrics:
          - type: accuracy
            value: 30.6
            name: 0-shot Accuracy
      - task:
          type: text-generation
          name: PIQA (Physical Intuition)
        dataset:
          name: PIQA
          type: ybisk/piqa
        metrics:
          - type: accuracy
            value: 54.7
            name: 0-shot Accuracy
---

# Brandon-Tiny-10M-Instruct

A 10.7M parameter instruction-following language model that punches far above its weight class. Small enough to run on a PlayStation 2's Emotion Engine. Named after a Cloudflare tunnel URL that sounded like a language model.

## Why This Exists

We wanted a language model small enough to run natively on a PlayStation 2 -- 32 MB of VRAM on the Emotion Engine, tested on both emulator and real hardware. Inspired by Karpathy's TinyStories, we looked for existing models small enough, and couldn't find anything decent. So we built one.

The name "Brandon Tiny" comes from a Cloudflare tunnel: one day we ran `cloudflared tunnel` and the random URL came back as something like `sugar-alaska-brandon-tiny.trycloudflare.com`. We were serving a custom agentic chat through it, and the URL looked so much like a language model name that we couldn't stop joking about it. The name stuck.

After 8 model variants, 3 architectural experiments, and one 3-phase training pipeline, Brandon Tiny turned out to be genuinely impressive for its size.

## Highlights

- **10.7M parameters** -- runs on a PS2 Emotion Engine (32 MB VRAM)
- **Beats our own 30M model** on fine-tuning loss (2.40 vs 2.61)
- **Zero repetition loops** across all generation tests
- **80% instruction following** on 25 diverse prompts
- **Trained in ~7 hours** on a single RTX 3090
- **3-phase pipeline**: Pretrain → Knowledge Distillation → Instruction Finetune

## Model Details

### Architecture

Llama 2 style decoder-only transformer with three modern enhancements:

| Spec | Value |
|------|-------|
| Parameters | 10,706,776 |
| Dimensions | 256 |
| Layers | 24 (12 unique with block sharing) |
| Attention Heads | 8 (2 KV heads, GQA 4:1) |
| FFN Hidden | 720 (SwiGLU) |
| Vocabulary | 8,192 (SentencePiece BPE) |
| Max Sequence | 512 tokens |
| Positional | RoPE (theta=10000) |
| Normalization | RMSNorm |
| Enhancements | DenseFormer + Value Residual + Register Tokens |

**Block Sharing (MobileLLM):** Adjacent layer pairs share weights, giving 24 effective layers from only 12 unique parameter blocks.

**DenseFormer:** Depth-Weighted Averaging connects all previous layer outputs to each subsequent layer.

**Value Residual:** Layer 0's value projection is added to all subsequent layers, preserving early representations.

**Register Tokens:** 4 learnable tokens prepended to input, acting as attention sinks.

### Training

3-phase pipeline, each phase addressing different aspects:

| Phase | Steps | LR Schedule | Data | Result |
|-------|-------|-------------|------|--------|
| 1. Foundation Pretrain | 15K | WSD (8e-4) | 600M tokens (Wiki 40% + SmolLM 30% + Synthetic 30%) | val_loss 4.39 |
| 2. Knowledge Distillation | 7.5K | WSD (4e-4) | Same pretrain data, 30M teacher | val_loss 4.84 |
| 3. Instruction Finetune | 12K | Cosine (2e-5) | 75K examples (chat + reasoning + replay) | **val_loss 2.40** |

**Anti-repetition training** in Phase 3: label smoothing (0.1) + unlikelihood training (0.5) + entropy regularization (0.01).

**Knowledge distillation** uses reverse KL divergence (mode-seeking) with temperature 2.0, following MiniPLM findings that reverse KLD is better for small students.

### Inference

| Metric | Value |
|--------|-------|
| Speed | 21 tokens/sec (RTX 3090) |
| VRAM | 51.5 MB allocated |
| Model size | 42.8 MB (fp32) / 21.4 MB (bf16) |

## Evaluation

### Standard Benchmarks (0-shot)

| Benchmark | Brandon-Tiny-10M | Random Baseline | Delta |
|-----------|:----------------:|:---------------:|:-----:|
| BLiMP (Grammar) | **73.3%** | 50.0% | +23.3 |
| HellaSwag (Commonsense) | **32.4%** | 25.0% | +7.4 |
| ARC-Easy (Science) | **30.6%** | 25.0% | +5.6 |
| PIQA (Physical Intuition) | **54.7%** | 50.0% | +4.7 |
| Winogrande (Coreference) | 50.3% | 50.0% | +0.3 |
| LAMBADA (Last Word) | **8.8%** | 0.0% | +8.8 |
| Wikitext-2 PPL | **224.2** | -- | -- |

### Custom Evaluation Suite

Tested against 7 models from our experimental series:

| Model | Params | Wikitext-2 PPL | Gen. Quality | Repetition | Instr. Following |
|-------|--------|:--------------:|:------------:|:----------:|:----------------:|
| **Brandon-Tiny-10M** | **10.7M** | **224.2** | **0.947** | **0.000** | **80%** |
| 10M Enhanced v2 | 10.7M | 329.2 | 0.939 | 0.000 | 80% |
| 30M v2 Original | 30.0M | 302.8 | 0.957 | 0.000 | 84% |
| 30M v2 Wiki | 30.0M | 168.6 | 0.918 | 0.000 | 72% |

Note: The 10M Optimal achieves comparable performance to 30M models with 3x fewer parameters.

## Usage

### Installation

```bash
git clone https://github.com/xaskasdf/brandon-tiny
cd brandon-tiny
pip install torch sentencepiece pyyaml numpy
```

### Inference

```python
from src.model import TinyLlama
from src.tokenizer import Tokenizer

# Load model
model = TinyLlama.from_checkpoint('checkpoints/10m_optimal/phase3_finetune/best.pt')
tokenizer = Tokenizer('data/tokenizer_8k.model')

# Chat format
prompt = "<|im_start|>user\nTell me a story about a cat.<|im_end|>\n<|im_start|>assistant\n"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor(tokens, device='cuda')

output = model.generate(
    input_ids,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    stop_tokens=tokenizer.get_stop_tokens(),
)

response = tokenizer.decode(output[0, len(tokens):].tolist())
print(response)
```

### Generation Examples

**Identity:**
> User: What is your name?
> Assistant: I'm Brandon Tiny, a small language model designed to help in limited environments.

**Creative:**
> User: Tell me a short story about a cat.
> Assistant: Once upon a time, there was an old but young cat named Charlie who lived in the small town of Pagima. Every day, he set out on adventures...

**Reasoning:**
> User: What happens when ice melts?
> Assistant: When ice causes a strong layer of water, the ice accumulates in the atmosphere due to the temperature and pressure...

## Limitations

- **Factual accuracy:** The model hallucinates freely. Do NOT use for factual question answering.
- **English only:** Trained exclusively on English data.
- **Short context:** Maximum 512 tokens.
- **No safety alignment:** No RLHF/DPO training has been applied.
- **Arithmetic:** Cannot reliably perform mathematical operations.
- **Vocabulary artifacts:** Occasionally produces garbled or invented words.
- **Standard benchmarks:** Expected to score near-random on MMLU, GSM8K, and other benchmarks designed for larger models.

## What This Model IS Good For

- Running a language model on a PlayStation 2 (or any device with 32+ MB VRAM)
- Research on ultra-small language models
- Understanding scaling behavior at extreme parameter budgets
- Edge/embedded deployment experiments
- Educational purposes (understanding LLM training pipelines)
- Prototyping conversational agents with minimal compute

## What This Model Is NOT Good For

- Factual question answering
- Mathematical reasoning
- Professional or production use
- Any task requiring reliability or safety guarantees

## Training Details

### Hardware
- Single NVIDIA RTX 3090 (24 GB VRAM)
- Total training time: ~7 hours across all 3 phases
- Operating system: Windows 11

### Data

**Pre-training (600M tokens):**
- [Wikipedia English](https://huggingface.co/datasets/wikimedia/wikipedia) (40%)
- [SmolLM Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (30%)
- Synthetic data generated by GPT-4o-mini (30%)

**Instruction fine-tuning (75,502 examples):**
- 57,000 curated chat instructions
- 19,944 reasoning/Chain-of-Thought examples
- ~200 pretrain replay examples (catastrophic forgetting mitigation)

**Evaluation datasets:**
- [BLiMP](https://huggingface.co/datasets/nyu-mll/blimp) - Grammatical knowledge
- [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) - Commonsense reasoning
- [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) - Science reasoning
- [PIQA](https://huggingface.co/datasets/ybisk/piqa) - Physical intuition
- [Winogrande](https://huggingface.co/datasets/allenai/winogrande) - Coreference resolution
- [LAMBADA](https://huggingface.co/datasets/cimec/lambada) - Last word prediction
- [Wikitext](https://huggingface.co/datasets/Salesforce/wikitext) - Perplexity evaluation

### Tokenizer
- SentencePiece BPE, 8,192 vocabulary
- Special tokens: `<|im_start|>`, `<|im_end|>`, `<|bos|>`, `<|eos|>`, `<|pad|>`
- Chat format: ChatML

### Training Hyperparameters

<details>
<summary>Phase 1: Foundation Pre-training</summary>

- Learning rate: 8e-4 → 8e-5 (WSD schedule)
- Steps: 15,000
- Batch size: 32 × 4 gradient accumulation = effective 65K tokens/step
- Warmup: 500 steps, Stable: 70%, Decay: 20%
- Weight decay: 0.1
- Optimizer: AdamW (beta1=0.9, beta2=0.95)
- Precision: bfloat16
</details>

<details>
<summary>Phase 2: Knowledge Distillation</summary>

- Teacher: Brandon-Tiny 30M v2 pretrained (30M params)
- Method: Reverse KL Divergence
- Temperature: 2.0
- Alpha: 0.5 (soft/hard target balance)
- Learning rate: 4e-4 → 4e-5 (WSD)
- Steps: 7,500
</details>

<details>
<summary>Phase 3: Instruction Fine-tuning</summary>

- Learning rate: 2e-5 → 2e-6 (cosine)
- Steps: 12,000
- Batch size: 16 × 4 gradient accumulation
- Weight decay: 0.01
- Label smoothing: 0.1
- Unlikelihood alpha: 0.5
- Entropy regularization beta: 0.01
- Loss masking: assistant tokens only
</details>

## Experimental Context

This model is the result of systematic exploration across 8 architectural variants. Key insight: **training methodology matters more than parameter count at this scale.** The 3-phase pipeline (Pretrain → Distill → Finetune) enabled a 10M model to outperform 30M models trained with a standard 2-phase approach.

For the full experimental report, see our [technical report](docs/technical_report.md).

## Links

- **Website:** [naranjositos.tech](https://naranjositos.tech/)
- **Code:** [github.com/xaskasdf/brandon-tiny](https://github.com/xaskasdf/brandon-tiny)
- **Technical Report:** [docs/technical_report.md](docs/technical_report.md)

## Citation

```bibtex
@misc{brandon-tiny-2026,
  title={Brandon-Tiny 10M: A 3-Phase Training Pipeline for Ultra-Small Instruction-Following Language Models},
  author={Samuel Cortes},
  year={2026},
  url={https://naranjositos.tech/}
}
```

## Acknowledgments

Architecture and techniques inspired by:
- MobileLLM (Meta, 2024) - Block sharing, deep-narrow design
- SmolLM2 (HuggingFace, 2025) - Data-centric small model training
- MiniCPM (2024) - WSD learning rate schedule
- DenseFormer (Pagliardini et al., 2024) - Depth-Weighted Averaging
- MiniPLM (ICLR 2025) - Reverse KLD for small model distillation
