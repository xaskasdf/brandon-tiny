# Brandon-Tiny 10M: A 3-Phase Training Pipeline for Ultra-Small Instruction-Following Language Models

**Authors:** Samuel Cortes
**Date:** February 2026
**Code:** https://github.com/[your-repo]/brandon-tiny

## Abstract

We present Brandon-Tiny 10M, a 10.7M parameter instruction-following language model that achieves competitive performance with models 3x its size through a novel 3-phase training pipeline combining foundation pre-training, knowledge distillation, and instruction fine-tuning. Built on a deep-narrow Llama 2 architecture enhanced with DenseFormer weighted averaging, Value Residual connections, and register tokens, our model demonstrates that careful training orchestration can compensate for limited parameter budgets. Trained entirely on a single RTX 3090 GPU, Brandon-Tiny 10M achieves a fine-tuning validation loss of 2.40 (surpassing our own 30M parameter models at 2.61), generates coherent text, follows instructions at 80% accuracy, and exhibits zero repetition loops across 20 free-generation tests. We describe our complete experimental journey across 8 model variants and analyze the key factors contributing to performance at this extreme scale.

## 1. Introduction

The dominant trend in language modeling has been scaling up: more parameters, more data, more compute. Yet practical deployment often demands the opposite -- models that run on edge devices, embedded systems, or resource-constrained environments. While research on efficient LLMs has flourished (MobileLLM, SmolLM2, Phi), most work targets the 125M-7B range. Models under 50M parameters remain largely unexplored for instruction-following tasks.

Our motivation was concrete: we wanted a language model capable of running natively on a PlayStation 2's Emotion Engine, which has only 32 MB of VRAM. Inspired by Karpathy's TinyStories, we searched for existing models small enough for this constraint and found none with instruction-following capabilities. The project's name -- Brandon Tiny -- originated from a Cloudflare tunnel URL (`sugar-alaska-brandon-tiny.trycloudflare.com`) generated while serving a custom agentic chat, which sounded so much like a language model name that it stuck.

We address this gap by systematically exploring what is achievable at 10.7M parameters. Our key contributions:

1. **A 3-phase training pipeline** (Pretrain → Distill → Finetune) specifically designed for ultra-small models, where each phase compensates for limitations of the previous one.
2. **Extensive ablation across 8 model variants**, revealing that training methodology matters more than raw parameter count at this scale.
3. **Anti-repetition training techniques** combining label smoothing, unlikelihood training, and entropy regularization that completely eliminate generation loops.
4. **Practical guidelines** for training sub-50M parameter instruction models on consumer hardware.

## 2. Architecture

### 2.1 Base Architecture

Brandon-Tiny 10M follows the Llama 2 architecture with modifications from MobileLLM (deep-narrow design with block sharing):

| Component | Specification |
|-----------|--------------|
| Dimensions | dim=256, hidden=720 |
| Layers | 24 (12 unique, block sharing) |
| Attention | 8 heads, 2 KV heads (GQA 4:1) |
| Vocabulary | 8,192 (SentencePiece BPE) |
| Max sequence | 512 tokens |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Position encoding | RoPE (theta=10000) |
| Total parameters | 10,706,776 |

### 2.2 Architectural Enhancements

Three enhancements from recent literature, which we validated on smaller experiments before applying to the final model:

**DenseFormer (Pagliardini et al., 2024):** Depth-Weighted Averaging (DWA) adds learnable weighted connections from all previous layer outputs to each subsequent layer, improving gradient flow in deep-narrow architectures.

**Value Residual Learning (Wang et al., 2024):** The value projection from layer 0 is added to all subsequent layers' value computations, preserving early representations that tend to be lost in deep networks.

**Register Tokens (Darcet et al., 2024):** 4 learnable tokens prepended to the input sequence act as "information sinks" for attention, reducing attention entropy collapse observed in small models.

### 2.3 MobileLLM Block Sharing

Adjacent pairs of layers share weights, effectively giving 12 unique parameter blocks applied twice. This halves the parameter count while maintaining depth, following the finding from MobileLLM that depth matters more than width for small models.

### 2.4 Design Rationale

The deep-narrow + block sharing design was chosen over standard proportional scaling based on our ablation studies:

| Model Variant | Architecture | Pretrain Loss | Finetune Loss |
|--------------|-------------|---------------|---------------|
| 10M v2 baseline | dim=192, 16 layers, sharing | 1.92 | 3.92 |
| 10M Enhanced v2 | dim=256, 24 layers, sharing + DWA + VR + Reg | 3.73 | 2.92 |

The enhanced architecture trades pretrain perplexity for dramatically better downstream performance, suggesting that the enhancements improve the model's ability to compress knowledge into limited parameters.

## 3. Training Pipeline

### 3.1 Overview

Our 3-phase pipeline is designed to address the core challenge of ultra-small models: they lack the capacity to learn everything from scratch. Each phase adds a different type of knowledge:

```
Phase 1: Foundation Pretrain    → Language modeling basics
Phase 2: Knowledge Distillation → Compressed knowledge from larger teacher
Phase 3: Instruction Finetune   → Task-specific behavior + anti-repetition
```

### 3.2 Phase 1: Foundation Pre-training

**Data:** Mixed corpus of 600M tokens:
- 40% [Wikipedia English](https://huggingface.co/datasets/wikimedia/wikipedia)
- 30% [SmolLM Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
- 30% Synthetic data (LLM-generated diverse topics)

**Configuration:**
- Learning rate: 8e-4 with WSD schedule (Warmup-Stable-Decay from MiniCPM)
- 15,000 steps, effective batch size 65,536 tokens/step
- Warmup: 500 steps, Stable: 70%, Decay: 20%
- Weight decay: 0.1, gradient clipping: 1.0
- Precision: bfloat16

**Result:** val_loss = 4.39, perplexity = 80.8

The WSD schedule from MiniCPM proved critical -- the decay phase consistently delivered the largest quality improvements in our experiments (often 0.3-0.5 loss reduction).

### 3.3 Phase 2: Knowledge Distillation

We distill from a pre-trained 30M parameter teacher (Brandon-Tiny 30M v2, val_loss 3.26) into our 10M student, representing a 2.8x compression ratio.

**Configuration:**
- Teacher: 30M v2 pretrained (dim=384, 28 layers, block sharing)
- Distillation method: Reverse KL Divergence (mode-seeking)
- Temperature: 2.0
- Alpha: 0.5 (equal weight hard/soft targets)
- Learning rate: 4e-4 with WSD schedule
- 7,500 steps

**Why Reverse KLD?** Following findings from MiniPLM (ICLR 2025), reverse KLD is mode-seeking rather than mean-seeking, which better suits small students that cannot cover the full distribution of the teacher. The student learns to focus on the teacher's most confident predictions rather than trying to approximate the entire distribution.

**Result:** val_loss = 4.84 (note: this is measured against hard targets; the model's soft-target loss was much lower)

### 3.4 Phase 3: Instruction Fine-tuning

**Data:** Merged instruction dataset (75,502 examples):
- 57,000 curated chat instructions (filtered from 70K original)
- 19,944 reasoning/CoT examples (math, logic, science)
- ~200 pretrain replay examples (text continuation tasks to prevent catastrophic forgetting)

**Chat Format:** ChatML with `<|im_start|>` / `<|im_end|>` tokens, loss computed only on assistant responses (mask_targets_only).

**Anti-repetition Training:**
- Label smoothing: 0.1
- Unlikelihood training (Welleck et al., 2020): alpha=0.5
- Entropy regularization: beta=0.01

This triple-technique approach was developed after observing severe repetition in early model versions. Each technique addresses a different failure mode: label smoothing prevents over-confident token prediction, unlikelihood training penalizes repeated n-grams during training, and entropy regularization maintains output diversity.

**Configuration:**
- Learning rate: 2e-5 with cosine schedule
- 12,000 steps
- Weight decay: 0.01
- Gradient clipping: 1.0

**Result:** val_loss = 2.3995 (new record across all models)

## 4. Experiments and Ablations

### 4.1 Model Variants

We trained 8 model variants to understand which factors most influence downstream performance:

| Model | Params | Architecture | Data | Pretrain Loss | Finetune Loss |
|-------|--------|-------------|------|---------------|---------------|
| 10M v2 baseline | 10.7M | Deep-narrow, sharing | TinyStories+SmolLM | 1.92 | 3.92 |
| 10M MTP | 10.7M | + Multi-Token Prediction | TinyStories+SmolLM | 2.10 | 3.45 |
| 10M Enhanced | 10.7M | + DWA + VR + Registers | Mixed | 3.73 | 2.92 |
| 10M Enhanced v2 | 10.7M | Same, more data | Mixed | 3.73 | 2.92 |
| 10M Dream | 10.7M | Ternary + Looped | Mixed | 3.81 | 2.98 |
| 10M Synthetic-only | 10.7M | Enhanced, synthetic pretrain | 100% synthetic | 1.96 | 3.62 |
| **10M Optimal** | **10.7M** | **Enhanced + 3-phase** | **Wiki+Mixed+Distill** | **4.39** | **2.40** |
| 30M v2 (reference) | 30.0M | Deep-narrow, sharing | TinyStories+SmolLM | 3.26 | 2.61 |

### 4.2 Key Findings

**Finding 1: Low pretrain loss ≠ good downstream performance.**
The synthetic-only model achieved the best pretrain loss (1.96) but one of the worst finetune losses (3.62). Conversely, the Optimal model had the highest pretrain loss (4.39) but the best finetune loss (2.40). This suggests that diversity and quality of pre-training data matters more than how well the model memorizes it.

**Finding 2: The 3-phase pipeline enables 10M to beat 30M.**
Our 10M Optimal (val_loss 2.40) significantly outperforms our best 30M model (val_loss 2.61), despite having 3x fewer parameters. The key difference is knowledge distillation + better data curation.

**Finding 3: Anti-repetition training is essential at this scale.**
Without the triple anti-repetition (label smoothing + unlikelihood + entropy reg), all models exhibited some degree of repetitive generation. With it, all tested models achieved zero sequence loops across 20 free-generation tests.

**Finding 4: Architectural enhancements compound.**
DenseFormer + Value Residual + Registers together improve finetune loss by ~1.0 point compared to the baseline architecture. Each contributes, but the combination is greater than the sum.

**Finding 5: Ternary quantization doesn't work at this scale.**
The Dream architecture (ternary weights + looped transformer) showed promise on paper but resulted in the worst instruction-following rate (28%) in benchmarks, with incoherent "word soup" generation despite reasonable loss values.

### 4.3 Benchmark Results

Automated evaluation across 6 models on our custom benchmark suite:

| Model | Wikitext-2 PPL | Generation Quality | Repetition | Instruction Following |
|-------|---------------|-------------------|------------|----------------------|
| **10M Optimal** | **224.2** | **0.947** | **0.000** | **80%** |
| 10M Enhanced v2 | 329.2 | 0.939 | 0.000 | 80% |
| 10M Synthetic-only | 818.0 | 0.984 | 0.000 | 96%* |
| 30M v2 Original | 302.8 | 0.957 | 0.000 | 84% |
| 30M v2 Wiki | 168.6 | 0.918 | 0.000 | 72% |
| 30M Dream | 1121.6 | 0.818 | 0.000 | 28% |

*Note: Synthetic-only's high instruction score is misleading -- verbose output accidentally matches keywords despite poor quality.

Standard academic benchmarks (0-shot, 1000 examples per task):

| Benchmark | Score | Random Baseline | Delta |
|-----------|-------|-----------------|-------|
| BLiMP (Grammar) | **73.3%** | 50.0% | +23.3 |
| HellaSwag (Commonsense) | **32.4%** | 25.0% | +7.4 |
| ARC-Easy (Science) | **30.6%** | 25.0% | +5.6 |
| PIQA (Physical Intuition) | **54.7%** | 50.0% | +4.7 |
| Winogrande (Coreference) | 50.3% | 50.0% | +0.3 |
| LAMBADA (Last Word) | **8.8%** | 0.0% | +8.8 |
| Wikitext-2 PPL | 224.2 | -- | -- |

BLiMP is the standout result, with several subtasks scoring above 90%: sentential negation NPI licensor (100%), existential there quantifiers (95.2%), wh-vs-that no gap long distance (95.1%). This demonstrates genuine grammatical knowledge acquired during training.

For comparison, the Super Tiny Language Models paper (2405.14159) reported their 50M parameter model (5x our size) scoring 25.6% on HellaSwag and 21% on ARC-Easy. Our 10.7M model at 32.4% and 30.6% respectively outperforms their 50M baseline on both benchmarks.

## 5. Qualitative Analysis

### 5.1 Generation Examples

**Identity awareness:**
```
User: What is your name?
Brandon-Tiny: I'm Brandon Tiny, a small language model designed to help in limited environments.
```

**Creative writing:**
```
User: Tell me a short story about a cat.
Brandon-Tiny: Once upon a time, there was an old but young cat named Charlie who lived
in the small town of Pagima. Every day, he set out on adventures...
```

**Factual (with hallucination):**
```
User: What is the capital of France?
Brandon-Tiny: The capital of the French Empire was Goé. [HALLUCINATION]
```

### 5.2 Failure Modes

At 10M parameters, the model exhibits:
- **Factual hallucination:** Confidently generates plausible but incorrect facts
- **Mathematical inability:** Cannot reliably perform arithmetic despite training on reasoning data
- **Vocabulary artifacts:** Occasionally produces garbled words ("cocooked", "tated")
- **Template pollution:** Some responses include training artifacts like LaTeX-style formatting

These are expected limitations at this parameter scale and are consistent with findings from the Super Tiny Language Models paper (2405.14159).

## 6. Infrastructure

### 6.1 Hardware

All training was performed on a single NVIDIA RTX 3090 (24GB VRAM):
- Phase 1: ~3 hours (15K steps)
- Phase 2: ~1.5 hours (7.5K steps)
- Phase 3: ~2 hours (12K steps)
- Total: ~6.5 hours wall clock time

### 6.2 Inference

- Speed: 21 tokens/second on RTX 3090
- VRAM: 51.5 MB allocated (211.8 MB reserved)
- Model size: 42.8 MB (fp32), 21.4 MB (bf16)

### 6.3 Tokenizer

Custom SentencePiece BPE tokenizer with 8,192 vocabulary, trained on a mix of TinyStories and SmolLM text. Includes ChatML special tokens (`<|im_start|>`, `<|im_end|>`, `<|bos|>`, `<|eos|>`, `<|pad|>`).

## 7. Related Work

**MobileLLM (Liu et al., 2024):** Introduced deep-narrow architectures and block sharing for sub-billion models. We adopt their block sharing strategy at 10M scale.

**SmolLM2 (Allal et al., 2025):** Data-centric approach to training small LMs. Their 135M model serves as our primary comparison point.

**Super Tiny Language Models (Galimzhanova et al., 2024):** Most directly comparable work targeting 10-100M parameters. Their 50M baseline scored near random on standard benchmarks (ARC-Easy 21%, HellaSwag 25.6%).

**DenseFormer (Pagliardini et al., 2024):** Depth-Weighted Averaging that we apply to improve gradient flow.

**Value Residual Learning (Wang et al., 2024):** Preserving early-layer value representations, particularly beneficial in deep networks.

**MiniCPM (Hu et al., 2024):** WSD learning rate schedule that we use for pre-training, providing consistent improvements in the decay phase.

**MiniPLM (ICLR 2025):** Showed reverse KLD is superior for distilling into small students. We adopt this for Phase 2.

## 8. Limitations

- **Not suitable for factual tasks:** The model hallucinates freely and should not be used for factual question answering.
- **English only:** Trained exclusively on English data.
- **Short context:** 512 token maximum sequence length.
- **No safety alignment:** No RLHF/DPO training; the model may generate inappropriate content.
- **Evaluation challenges:** Standard LLM benchmarks (MMLU, GSM8K) are meaningless at this scale; custom evaluation is required.

## 9. Conclusion

Brandon-Tiny 10M demonstrates that with careful architecture design, multi-phase training, and knowledge distillation, a 10.7M parameter model can achieve instruction-following capabilities previously associated with much larger models. Our 3-phase pipeline (Pretrain → Distill → Finetune) is the key innovation, enabling the model to outperform a 30M parameter counterpart on fine-tuning loss. The complete training pipeline runs in under 7 hours on a single consumer GPU, making this approach accessible for research and experimentation.

Future work includes: (1) DPO/GRPO alignment training, (2) scaling the 3-phase pipeline to our 110M model, and (3) quantization for true edge deployment.

## 10. Reproducibility

All code, configurations, and training scripts are available at [repository]. The complete training pipeline can be run with:

```bash
python scripts/train_10m_optimal.py
```

Model checkpoints and the tokenizer will be released on HuggingFace.

## References

[Full bibliography to be added with proper citations]
