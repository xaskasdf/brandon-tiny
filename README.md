# Brandon-Tiny

Ultra-small instruction-following language models (10M-110M parameters) that run on extreme hardware constraints -- including a PlayStation 2.

**Website:** [naranjositos.tech](https://naranjositos.tech/)
**Model on HuggingFace:** [brandon-tiny-10m-instruct](https://huggingface.co/xaskasdf/brandon-tiny-10m-instruct)
**Datasets on HuggingFace:** [brandon-tiny-pretrain](https://huggingface.co/datasets/xaskasdf/brandon-tiny-pretrain) | [brandon-tiny-instruct](https://huggingface.co/datasets/xaskasdf/brandon-tiny-instruct)
**Technical Report:** [GitHub Pages](https://xaskasdf.github.io/brandon-tiny/paper.html) | [PDF](https://factory.naranjositos.tech/downloads/brandon_tiny_10m.pdf)

## The Story

We wanted a language model that could run natively on a PlayStation 2's Emotion Engine (32 MB VRAM). Inspired by Karpathy's TinyStories, we searched for existing models small enough and couldn't find anything decent. So we built one.

The name comes from a Cloudflare tunnel URL (`sugar-alaska-brandon-tiny.trycloudflare.com`) generated while serving a custom agentic chat. It sounded like a language model name, and after enough jokes, it stuck.

## Results

The flagship **10M Optimal** model achieves:

| Benchmark | Score | Random |
|-----------|:-----:|:------:|
| BLiMP (Grammar) | **73.3%** | 50.0% |
| HellaSwag | **32.4%** | 25.0% |
| ARC-Easy | **30.6%** | 25.0% |
| PIQA | **54.7%** | 50.0% |
| Wikitext-2 PPL | **224.2** | -- |

For context: the Super Tiny Language Models paper's 50M model (5x our size) scored 25.6% on HellaSwag and 21% on ARC-Easy. Our 10.7M model beats both.

The model generates coherent text, knows its own name, follows instructions at 80%, and never gets stuck in repetition loops. All trained in ~7 hours on a single RTX 3090.

## Architecture

Llama 2 decoder-only transformer, deep-narrow (MobileLLM style):

- **10.7M parameters** (dim=256, 24 layers with block sharing)
- DenseFormer + Value Residual + Register Tokens
- 8K BPE tokenizer, 512 max sequence, ChatML format
- Model size: 42.8 MB (fp32) / 21.4 MB (bf16)

## 3-Phase Training Pipeline

The key innovation is a 3-phase pipeline where each phase compensates for the previous:

```
Phase 1: Foundation Pretrain     (15K steps, WSD schedule)
    └─ Language modeling on Wiki + SmolLM + Synthetic data

Phase 2: Knowledge Distillation  (7.5K steps, reverse KLD)
    └─ Learn compressed knowledge from 30M teacher

Phase 3: Instruction Finetune    (12K steps, cosine + anti-repetition)
    └─ 75K examples: chat + reasoning + pretrain replay
```

## Quick Start

```bash
# Install
pip install torch sentencepiece pyyaml numpy datasets

# Generate text
python scripts/chat.py --checkpoint checkpoints/10m_optimal/phase3_finetune/best.pt

# Run the full 3-phase training pipeline
python scripts/train_10m_optimal.py

# Run benchmarks
python scripts/benchmark_serious.py
python scripts/benchmark_10m.py --all-models
```

## Project Structure

```
brandon-tiny/
├── src/
│   ├── model.py              # TinyLlama architecture (Llama 2 + enhancements)
│   ├── tokenizer.py          # SentencePiece wrapper with ChatML support
│   ├── training/
│   │   ├── trainer.py        # Training loop, WSD/cosine schedules, anti-repetition
│   │   ├── pretrain.py       # Pre-training on text corpora
│   │   ├── finetune.py       # Instruction fine-tuning
│   │   └── distill.py        # Knowledge distillation (forward/reverse KLD)
│   └── data/
│       ├── dataset.py        # Pre-training dataset (tokenized .pkl)
│       └── chat_dataset.py   # Instruction dataset (JSONL, ChatML)
├── configs/                   # YAML configs for all model variants
├── scripts/
│   ├── train_10m_optimal.py  # Full 3-phase training pipeline
│   ├── benchmark_serious.py  # Academic benchmarks (BLiMP, HellaSwag, etc.)
│   ├── benchmark_10m.py      # Custom evaluation suite
│   ├── download_datasets.py  # Download training data
│   └── chat.py               # Interactive chat demo
├── docs/
│   ├── technical_report.md   # Full technical report / paper
│   └── MODEL_CARD.md         # HuggingFace model card
└── data/
    └── tokenizer_8k.model    # SentencePiece tokenizer (8192 vocab)
```

## Model Variants

We trained 8 variants to understand what works at this scale:

| Model | Params | Pretrain Loss | Finetune Loss | Notes |
|-------|--------|:------------:|:------------:|-------|
| **10M Optimal** | 10.7M | 4.39 | **2.40** | 3-phase pipeline, best overall |
| 10M Enhanced v2 | 10.7M | 3.73 | 2.92 | DenseFormer+VR+Registers |
| 10M Dream | 10.7M | 3.81 | 2.98 | Ternary weights (failed) |
| 10M Synthetic-only | 10.7M | 1.96 | 3.62 | Proved synthetic data = poor transfer |
| 30M v2 Original | 30.0M | 3.26 | 2.61 | Best 30M, but 10M Optimal beats it |
| 30M v2 Wiki | 30.0M | 4.03 | 2.80 | Wikipedia pretrain |
| 30M Dream | 31.1M | 2.97 | 4.22 | Ternary, worst overall |

Key finding: **training methodology > parameter count.** The 10M Optimal (val_loss 2.40) outperforms all 30M models (best: 2.61).

## Datasets

**Pre-training (600M tokens):**
- [Wikipedia English](https://huggingface.co/datasets/wikimedia/wikipedia) (40%)
- [SmolLM Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (30%)
- Synthetic data via GPT-4o-mini (30%)

**Instruction fine-tuning (75K examples):**
- Curated chat instructions + reasoning/CoT examples

**Evaluation:**
[BLiMP](https://huggingface.co/datasets/nyu-mll/blimp) | [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) | [ARC](https://huggingface.co/datasets/allenai/ai2_arc) | [PIQA](https://huggingface.co/datasets/ybisk/piqa) | [Winogrande](https://huggingface.co/datasets/allenai/winogrande) | [LAMBADA](https://huggingface.co/datasets/cimec/lambada) | [Wikitext](https://huggingface.co/datasets/Salesforce/wikitext)

## Hardware

- **Training:** Single NVIDIA RTX 3090 (24 GB)
- **Inference:** 21 tokens/sec, 51 MB VRAM
- **Target:** PlayStation 2 Emotion Engine (32 MB VRAM)

## Citation

```bibtex
@misc{brandon-tiny-2026,
  title={Brandon-Tiny 10M: A 3-Phase Training Pipeline for Ultra-Small Instruction-Following Language Models},
  author={Samuel Cortes},
  year={2026},
  url={https://naranjositos.tech/}
}
```

## License

Apache 2.0
