#!/usr/bin/env python
"""
train_10m_optimal.py - Optimal 10M Model Training Pipeline
============================================================

Goal: Create the best possible 10M parameter language model by combining
every technique proven effective in Brandon-Tiny experiments.

EMPIRICAL INSIGHTS:
  1. Data diversity >> synthetic purity
     - Synthetic-only: pretrain 1.96 -> finetune 3.63 (poor transfer)
     - Enhanced v2:    pretrain 3.73 -> finetune 2.92 (good transfer)
     Conclusion: Real data diversity is critical for downstream quality

  2. Wikipedia is untapped gold for 10M
     - Dramatically improved 30M and 110M models
     - Never tried for 10M -> biggest opportunity

  3. Knowledge distillation transfers capability
     - 30M v2 teacher available (finetune val_loss 2.61)
     - Reverse KLD proven better for small students (MINIPLM, ICLR 2025)

  4. 19.9K reasoning CoT examples sitting unused
     - Adds question diversity and structured answer patterns

  5. Anti-repetition + enhanced architecture = consistent wins

PIPELINE:
  Phase 1: Foundation Pretrain (~15K steps)
    Data: combined_30m_v2 (40% Wiki + 30% SmolLM + 30% Synthetic)
    Schedule: WSD (proven superior to cosine for pretrain)
    Why: First time adding Wikipedia knowledge to a 10M model

  Phase 2: Knowledge Distillation (~7.5K steps)
    Teacher: 30M v2 pretrained (3x compression ratio)
    Loss: 50% reverse-KLD + 50% CE (mode-seeking for small student)
    Why: Transfer structured representations from a proven larger model

  Phase 3: Instruction Finetune (~12K steps)
    Data: chat_curated (57K) + reasoning CoT (19.9K) + pretrain replay (8K)
    Techniques: label_smoothing, unlikelihood training, entropy regularization
    Why: Maximum instruction diversity with catastrophic forgetting prevention

Usage:
    python scripts/train_10m_optimal.py                    # Full pipeline
    python scripts/train_10m_optimal.py --start-phase 2    # Resume from distillation
    python scripts/train_10m_optimal.py --start-phase 3    # Resume from finetune
    python scripts/train_10m_optimal.py --eval-only        # Just evaluate best model
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import TinyLlama, ModelConfig
from src.tokenizer import Tokenizer
from src.data.dataset import TextFileDataset, create_dataloader
from src.data.chat_dataset import InstructionDataset, collate_chat
from src.training.trainer import Trainer, TrainingConfig, get_lr
from src.training.distill import Distiller


# ============================================================
# CONFIGURATION - Every choice here is backed by experiment data
# ============================================================

# Architecture: 10M Enhanced (proven best architecture for 10M scale)
# DenseFormer + Value Residual + Registers = consistent wins across all models
MODEL = dict(
    dim=256, n_layers=24, n_heads=8, n_kv_heads=2,
    vocab_size=8192, hidden_dim=720, max_seq_len=512,
    dropout=0.05, weight_tying=True, norm_eps=1e-5,
    rope_theta=10000.0, block_sharing=True,
    dense_former=True, value_residual=True, n_registers=4,
)

OUTPUT_DIR = Path('checkpoints/10m_optimal')
TOKENIZER_PATH = 'data/tokenizer_8k.model'

# Data paths
PRETRAIN_DATA = 'data/combined_30m_v2'           # 600M tokens: 40% Wiki + 30% SmolLM + 30% Synthetic
FINETUNE_DATA = 'data/chat_curated'              # 57K curated instruction examples
REASONING_DATA = 'D:/AI/datasets/reasoning/combined_reasoning.json'  # 19.9K CoT examples
TEACHER_CHECKPOINT = 'checkpoints/30m_v2/pretrain/best.pt'           # 30M teacher

# Merged finetune output
MERGED_FINETUNE_DIR = Path('data/finetune_10m_optimal')


# ============================================================
# PHASE CONFIGURATIONS
# ============================================================

# Phase 1: Foundation Pretrain
# - Wikipedia for real-world knowledge (never tried at 10M!)
# - WSD schedule (MiniCPM: decay phase gives biggest quality jumps)
# - 15K steps ~ 1B tokens with data looping (small models benefit from repetition)
PHASE1 = dict(
    batch_size=32,
    gradient_accumulation_steps=4,   # effective batch: 32*4*512 = 65K tokens/step
    learning_rate=8e-4,
    min_lr=8e-5,
    warmup_iters=500,
    max_iters=15000,
    lr_schedule='wsd',
    stable_frac=0.7,
    decay_frac=0.2,
    weight_decay=0.1,
    eval_interval=500,
    eval_iters=100,
    save_interval=2000,
)

# Phase 2: Knowledge Distillation
# - 30M v2 teacher (3x larger, pretrain val_loss 3.26)
# - Reverse KLD: mode-seeking, proven better for small students
# - Smaller batch (teacher + student both in VRAM, 30M=120MB + 10M=43MB = fine)
# - Lower LR: we're refining, not learning from scratch
PHASE2 = dict(
    batch_size=24,
    gradient_accumulation_steps=4,   # effective batch: 24*4*512 = 49K tokens/step
    learning_rate=4e-4,
    min_lr=4e-5,
    warmup_iters=300,
    max_iters=7500,
    lr_schedule='wsd',
    stable_frac=0.7,
    decay_frac=0.2,
    weight_decay=0.1,
    eval_interval=500,
    eval_iters=100,
    save_interval=1500,
    alpha=0.5,           # 50% distillation + 50% CE
    temperature=2.0,     # Soft targets temperature
    reverse_kld=True,    # Mode-seeking (MINIPLM): student focuses on main teacher modes
)

# Phase 3: Instruction Finetune
# - Merged data: chat (57K) + reasoning CoT (19.9K) + pretrain replay (~8K)
# - Anti-repetition: all three techniques that proved effective
# - Cosine schedule: smooth decay for finetune stability
PHASE3 = dict(
    batch_size=16,
    gradient_accumulation_steps=4,   # effective batch: 16*4*512 = 32K tokens/step
    learning_rate=2e-5,
    min_lr=2e-6,
    warmup_iters=300,
    max_iters=12000,
    lr_schedule='cosine',
    weight_decay=0.01,
    eval_interval=500,
    eval_iters=50,
    save_interval=1000,
    label_smoothing=0.1,       # Prevents over-confident predictions
    unlikelihood_alpha=0.5,    # Penalizes repeated tokens
    entropy_reg_beta=0.01,     # Keeps output distribution diverse
)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_finetune_data(tokenizer):
    """Prepare merged finetune dataset: chat_curated + reasoning + pretrain replay.

    This is the secret sauce: combining three diverse data sources gives
    the model broad instruction-following capability while preventing
    catastrophic forgetting of pretrain knowledge.
    """
    MERGED_FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    train_out = MERGED_FINETUNE_DIR / 'train.jsonl'
    val_out = MERGED_FINETUNE_DIR / 'val.jsonl'

    if train_out.exists() and val_out.exists():
        with open(train_out) as f:
            n_train = sum(1 for _ in f)
        with open(val_out) as f:
            n_val = sum(1 for _ in f)
        print(f"Finetune data exists: {n_train:,} train, {n_val:,} val")
        return

    all_train = []
    all_val = []

    # --- Source 1: Chat curated (proven quality, 57K examples) ---
    print("Loading chat_curated data...")
    chat_train_path = Path(FINETUNE_DATA) / 'train.jsonl'
    chat_val_path = Path(FINETUNE_DATA) / 'val.jsonl'

    with open(chat_train_path, encoding='utf-8') as f:
        chat_train = [json.loads(line) for line in f if line.strip()]
    with open(chat_val_path, encoding='utf-8') as f:
        chat_val = [json.loads(line) for line in f if line.strip()]

    all_train.extend(chat_train)
    all_val.extend(chat_val)
    print(f"  Chat curated: {len(chat_train):,} train, {len(chat_val):,} val")

    # --- Source 2: Reasoning CoT (19.9K examples, NEVER USED BEFORE) ---
    # These add diverse questions and structured reasoning patterns.
    # For 512 context, we truncate long reasoning but keep the approach + answer.
    print("Loading reasoning CoT data...")
    if Path(REASONING_DATA).exists():
        with open(REASONING_DATA, encoding='utf-8') as f:
            reasoning_raw = json.load(f)

        reasoning_examples = []
        skipped = 0
        for item in reasoning_raw:
            user = item.get('user', '').strip()
            assistant = item.get('assistant', '').strip()
            reasoning = item.get('reasoning', '').strip()

            if not user or (not assistant and not reasoning):
                skipped += 1
                continue

            # Budget: ~512 tokens = ~2000 chars total
            # ChatML formatting + system prompt + instruction ~ 400 chars
            # Leaves ~1600 chars for the output
            max_output_chars = 1400

            # Strategy: include reasoning if it fits, otherwise just answer
            if reasoning and assistant and reasoning != assistant:
                full_output = reasoning + "\n\n" + assistant
                if len(full_output) <= max_output_chars:
                    output = full_output
                elif len(assistant) <= max_output_chars:
                    output = assistant
                else:
                    output = assistant[:max_output_chars]
            elif reasoning:
                output = reasoning[:max_output_chars]
            else:
                output = assistant[:max_output_chars]

            reasoning_examples.append({
                'instruction': user,
                'input': '',
                'output': output
            })

        # 90/10 train/val split
        random.seed(42)
        random.shuffle(reasoning_examples)
        split = int(len(reasoning_examples) * 0.9)
        reasoning_train = reasoning_examples[:split]
        reasoning_val = reasoning_examples[split:]

        all_train.extend(reasoning_train)
        all_val.extend(reasoning_val)
        print(f"  Reasoning CoT: {len(reasoning_train):,} train, {len(reasoning_val):,} val "
              f"(skipped {skipped})")
    else:
        print(f"  WARNING: Reasoning data not found at {REASONING_DATA}, skipping")

    # --- Source 3: Pretrain Replay (~10% of total) ---
    # Mix text-continuation examples to prevent catastrophic forgetting.
    # The model keeps seeing diverse text patterns while learning instructions.
    print("Creating pretrain replay examples...")
    replay_target = int(len(all_train) * 0.10)
    replay_examples = create_pretrain_replay(tokenizer, replay_target)
    all_train.extend(replay_examples)
    print(f"  Pretrain replay: {len(replay_examples):,} examples")

    # Shuffle everything to mix sources
    random.seed(42)
    random.shuffle(all_train)
    random.shuffle(all_val)

    # Save merged data
    with open(train_out, 'w', encoding='utf-8') as f:
        for item in all_train:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(val_out, 'w', encoding='utf-8') as f:
        for item in all_val:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n  Merged finetune data saved:")
    print(f"    Train: {len(all_train):,} examples -> {train_out}")
    print(f"    Val:   {len(all_val):,} examples -> {val_out}")


def create_pretrain_replay(tokenizer, count, chunk_tokens=200):
    """Create text-continuation examples from pretrain data.

    Samples random passages, splits into prompt/continuation, and formats
    as instructions ("Continue this text:" -> continuation).

    This is a simple but effective technique to prevent catastrophic forgetting:
    the model keeps seeing diverse text patterns during instruction tuning.
    """
    pretrain_path = Path(PRETRAIN_DATA) / 'train_tokens.npy'
    if not pretrain_path.exists():
        print(f"    Warning: {pretrain_path} not found, no replay examples")
        return []

    tokens = np.load(pretrain_path, mmap_mode='r')
    total_tokens = len(tokens)

    templates = [
        "Continue writing the following text:",
        "Complete the following passage:",
        "Write a continuation of this text:",
        "Continue this:",
        "Finish writing this passage:",
    ]

    examples = []
    random.seed(123)
    attempts = 0
    max_attempts = count * 5  # Generate extra, filter for quality

    while len(examples) < count and attempts < max_attempts:
        attempts += 1

        # Random position in the token stream
        start = random.randint(0, total_tokens - chunk_tokens * 2)
        chunk = tokens[start:start + chunk_tokens * 2]

        # Decode to text
        text = tokenizer.decode(chunk.tolist())

        # Split at a sentence boundary near the middle
        sentences = text.split('. ')
        if len(sentences) < 3:
            continue

        mid = len(sentences) // 2
        prompt_text = '. '.join(sentences[:mid]).strip()
        continuation = '. '.join(sentences[mid:]).strip()

        # Add period back if missing
        if prompt_text and not prompt_text.endswith('.'):
            prompt_text += '.'

        # Quality filters: meaningful chunks, not too short or long
        if len(prompt_text) < 50 or len(continuation) < 50:
            continue
        if len(prompt_text) > 600 or len(continuation) > 600:
            continue
        # Skip if mostly special chars or numbers
        alpha_ratio = sum(c.isalpha() for c in prompt_text) / max(len(prompt_text), 1)
        if alpha_ratio < 0.5:
            continue

        examples.append({
            'instruction': random.choice(templates),
            'input': prompt_text,
            'output': continuation
        })

    return examples[:count]


# ============================================================
# PHASE 1: FOUNDATION PRETRAIN
# ============================================================

def run_phase1(device):
    """Foundation pretrain on Wikipedia-enriched data.

    This is the first time we use Wikipedia for a 10M model.
    The data mix (40% Wiki + 30% SmolLM + 30% Synthetic) provides:
    - Real-world knowledge and facts (Wikipedia)
    - Diverse writing styles and web text (SmolLM)
    - Clean, structured patterns (Synthetic)
    """
    phase_dir = OUTPUT_DIR / 'phase1_pretrain'

    # Check if already complete
    final_path = phase_dir / 'final.pt'
    if final_path.exists():
        print(f"\n{'='*60}")
        print(f"Phase 1 already complete: {final_path}")
        print(f"{'='*60}")
        return phase_dir / 'best.pt'

    print(f"\n{'='*60}")
    print("PHASE 1: FOUNDATION PRETRAIN")
    print(f"  Data: {PRETRAIN_DATA}")
    print(f"  Mix: 40% Wikipedia + 30% SmolLM + 30% Synthetic")
    print(f"  Steps: {PHASE1['max_iters']:,} | Schedule: WSD")
    print(f"  Tokens/step: {PHASE1['batch_size'] * PHASE1['gradient_accumulation_steps'] * MODEL['max_seq_len']:,}")
    print(f"{'='*60}\n")

    tokenizer = Tokenizer(TOKENIZER_PATH)

    # Create model from scratch
    model_config = ModelConfig(**MODEL)
    model_config.vocab_size = tokenizer.vocab_size
    model = TinyLlama(model_config)
    print(f"Model: {model.count_parameters():,} parameters")

    # Load data
    train_dataset = TextFileDataset(PRETRAIN_DATA, tokenizer, MODEL['max_seq_len'], split='train')
    val_dataset = TextFileDataset(PRETRAIN_DATA, tokenizer, MODEL['max_seq_len'], split='validation')

    train_loader = create_dataloader(
        train_dataset, PHASE1['batch_size'], shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = create_dataloader(
        val_dataset, PHASE1['batch_size'], shuffle=False, num_workers=0, drop_last=False
    )

    # Training config
    config = TrainingConfig(
        learning_rate=PHASE1['learning_rate'],
        min_lr=PHASE1['min_lr'],
        warmup_iters=PHASE1['warmup_iters'],
        max_iters=PHASE1['max_iters'],
        lr_schedule=PHASE1['lr_schedule'],
        stable_frac=PHASE1['stable_frac'],
        decay_frac=PHASE1['decay_frac'],
        weight_decay=PHASE1['weight_decay'],
        batch_size=PHASE1['batch_size'],
        gradient_accumulation_steps=PHASE1['gradient_accumulation_steps'],
        eval_interval=PHASE1['eval_interval'],
        eval_iters=PHASE1['eval_iters'],
        save_interval=PHASE1['save_interval'],
        log_interval=10,
        dtype='bfloat16',
        output_dir=str(phase_dir),
    )

    trainer = Trainer(model, train_loader, val_loader, config, device=device)

    # Resume from latest checkpoint if available
    resume_path = None
    if phase_dir.exists():
        step_files = sorted(phase_dir.glob('step_*.pt'))
        if step_files:
            resume_path = str(step_files[-1])
            print(f"Resuming from {resume_path}")

    trainer.train(resume_from=resume_path)

    # Report
    print(f"\nPhase 1 complete! Best val_loss: {trainer.best_val_loss:.4f}")
    print(f"Perplexity: {math.exp(trainer.best_val_loss):.2f}")
    return phase_dir / 'best.pt'


# ============================================================
# PHASE 2: KNOWLEDGE DISTILLATION
# ============================================================

def run_phase2(device, student_checkpoint):
    """Knowledge distillation from 30M v2 teacher.

    The 30M v2 model (pretrain val_loss 3.26) has 3x more parameters and
    learned richer representations. Distillation transfers this knowledge
    to our 10M student using soft targets.

    Reverse KLD (mode-seeking) is used because small students with limited
    capacity should focus on the main modes of the teacher distribution
    rather than trying to cover all of it (which wastes capacity).
    """
    phase_dir = OUTPUT_DIR / 'phase2_distill'

    # Check if already complete
    final_path = phase_dir / 'final.pt'
    if final_path.exists():
        print(f"\n{'='*60}")
        print(f"Phase 2 already complete: {final_path}")
        print(f"{'='*60}")
        return phase_dir / 'best.pt'

    # Check teacher exists
    if not Path(TEACHER_CHECKPOINT).exists():
        print(f"\nWARNING: Teacher not found at {TEACHER_CHECKPOINT}")
        print("Skipping Phase 2 (distillation). Using Phase 1 checkpoint directly.")
        return student_checkpoint

    print(f"\n{'='*60}")
    print("PHASE 2: KNOWLEDGE DISTILLATION")
    print(f"  Teacher: {TEACHER_CHECKPOINT}")
    print(f"  Student: {student_checkpoint}")
    print(f"  KLD: Reverse (mode-seeking) | Alpha: {PHASE2['alpha']} | T: {PHASE2['temperature']}")
    print(f"  Steps: {PHASE2['max_iters']:,} | Schedule: WSD")
    print(f"{'='*60}\n")

    tokenizer = Tokenizer(TOKENIZER_PATH)

    # Load teacher (frozen during training)
    print("Loading teacher model...")
    teacher = TinyLlama.from_checkpoint(TEACHER_CHECKPOINT, device=device)
    teacher.eval()
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {t_params:,} params")

    # Load student from Phase 1
    print("Loading student model...")
    student = TinyLlama.from_checkpoint(str(student_checkpoint), device='cpu')
    s_params = sum(p.numel() for p in student.parameters())
    print(f"  Student: {s_params:,} params")
    print(f"  Compression: {t_params / s_params:.1f}x")

    # Same pretrain data (teacher provides the additional learning signal)
    train_dataset = TextFileDataset(PRETRAIN_DATA, tokenizer, MODEL['max_seq_len'], split='train')
    val_dataset = TextFileDataset(PRETRAIN_DATA, tokenizer, MODEL['max_seq_len'], split='validation')

    train_loader = create_dataloader(
        train_dataset, PHASE2['batch_size'], shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = create_dataloader(
        val_dataset, PHASE2['batch_size'], shuffle=False, num_workers=0, drop_last=False
    )

    # Training config
    config = TrainingConfig(
        learning_rate=PHASE2['learning_rate'],
        min_lr=PHASE2['min_lr'],
        warmup_iters=PHASE2['warmup_iters'],
        max_iters=PHASE2['max_iters'],
        lr_schedule=PHASE2['lr_schedule'],
        stable_frac=PHASE2['stable_frac'],
        decay_frac=PHASE2['decay_frac'],
        weight_decay=PHASE2['weight_decay'],
        batch_size=PHASE2['batch_size'],
        gradient_accumulation_steps=PHASE2['gradient_accumulation_steps'],
        eval_interval=PHASE2['eval_interval'],
        eval_iters=PHASE2['eval_iters'],
        save_interval=PHASE2['save_interval'],
        log_interval=10,
        dtype='bfloat16',
        output_dir=str(phase_dir),
    )

    # Run distillation
    distiller = Distiller(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        alpha=PHASE2['alpha'],
        temperature=PHASE2['temperature'],
        reverse_kld=PHASE2['reverse_kld'],
        device=device,
    )

    distiller.train()

    # Free teacher VRAM
    del teacher
    torch.cuda.empty_cache()

    print(f"\nPhase 2 complete! Best val_loss: {distiller.best_val_loss:.4f}")
    return phase_dir / 'best.pt'


# ============================================================
# PHASE 3: INSTRUCTION FINETUNE
# ============================================================

def run_phase3(device, student_checkpoint):
    """Instruction finetune with reasoning data + pretrain replay.

    Three data sources combined for maximum instruction diversity:
    1. Chat curated (57K) - proven high-quality instruction pairs
    2. Reasoning CoT (19.9K) - adds chain-of-thought capability (FIRST TIME!)
    3. Pretrain replay (~8K) - prevents catastrophic forgetting

    Anti-repetition arsenal:
    - Label smoothing (0.1): prevents over-confident token predictions
    - Unlikelihood training (0.5): explicitly penalizes repeated tokens
    - Entropy regularization (0.01): keeps output distribution diverse
    - Inference: repetition_penalty=1.2 + no_repeat_ngram_size=3
    """
    phase_dir = OUTPUT_DIR / 'phase3_finetune'

    # Check if already complete
    final_path = phase_dir / 'final.pt'
    if final_path.exists():
        print(f"\n{'='*60}")
        print(f"Phase 3 already complete: {final_path}")
        print(f"{'='*60}")
        return phase_dir / 'best.pt'

    print(f"\n{'='*60}")
    print("PHASE 3: INSTRUCTION FINETUNE")
    print(f"  Data: chat_curated + reasoning_CoT + pretrain_replay")
    print(f"  Model: {student_checkpoint}")
    print(f"  Steps: {PHASE3['max_iters']:,} | Schedule: Cosine")
    print(f"  Anti-rep: LS={PHASE3['label_smoothing']}, "
          f"UL={PHASE3['unlikelihood_alpha']}, ER={PHASE3['entropy_reg_beta']}")
    print(f"{'='*60}\n")

    tokenizer = Tokenizer(TOKENIZER_PATH)

    # Prepare merged finetune data (only runs once, then cached)
    prepare_finetune_data(tokenizer)

    # Load model from previous phase
    print(f"Loading model from {student_checkpoint}...")
    model = TinyLlama.from_checkpoint(str(student_checkpoint), device='cpu')
    print(f"Model: {model.count_parameters():,} params")

    # Load merged datasets
    train_dataset = InstructionDataset(
        data_path=str(MERGED_FINETUNE_DIR / 'train.jsonl'),
        tokenizer=tokenizer,
        max_seq_len=MODEL['max_seq_len'],
        system_prompt="You are a helpful assistant.",
        mask_targets_only=True,
    )
    val_dataset = InstructionDataset(
        data_path=str(MERGED_FINETUNE_DIR / 'val.jsonl'),
        tokenizer=tokenizer,
        max_seq_len=MODEL['max_seq_len'],
        system_prompt="You are a helpful assistant.",
        mask_targets_only=True,
    )

    collate_fn = partial(collate_chat, pad_id=tokenizer.pad_id or 0)

    train_loader = DataLoader(
        train_dataset, batch_size=PHASE3['batch_size'],
        shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=PHASE3['batch_size'],
        shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False
    )

    print(f"Train: {len(train_dataset):,} examples")
    print(f"Val: {len(val_dataset):,} examples")

    # Training config with anti-repetition
    config = TrainingConfig(
        learning_rate=PHASE3['learning_rate'],
        min_lr=PHASE3['min_lr'],
        warmup_iters=PHASE3['warmup_iters'],
        max_iters=PHASE3['max_iters'],
        lr_schedule=PHASE3['lr_schedule'],
        weight_decay=PHASE3['weight_decay'],
        batch_size=PHASE3['batch_size'],
        gradient_accumulation_steps=PHASE3['gradient_accumulation_steps'],
        eval_interval=PHASE3['eval_interval'],
        eval_iters=PHASE3['eval_iters'],
        save_interval=PHASE3['save_interval'],
        log_interval=10,
        dtype='bfloat16',
        label_smoothing=PHASE3['label_smoothing'],
        unlikelihood_alpha=PHASE3['unlikelihood_alpha'],
        entropy_reg_beta=PHASE3['entropy_reg_beta'],
        output_dir=str(phase_dir),
    )

    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    trainer.train()

    print(f"\nPhase 3 complete! Best val_loss: {trainer.best_val_loss:.4f}")
    return phase_dir / 'best.pt'


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(device, checkpoint_path):
    """Generate sample responses to evaluate the final model."""
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"Model: {checkpoint_path}")
    print(f"{'='*60}\n")

    tokenizer = Tokenizer(TOKENIZER_PATH)
    model = TinyLlama.from_checkpoint(str(checkpoint_path), device=device)
    model.eval()

    test_prompts = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Explain what photosynthesis is in simple terms.",
        "Write a short poem about the moon.",
        "Who are you?",
        "Why is the sky blue?",
        "What is machine learning?",
        "If I have 10 apples and give away 3, how many do I have?",
    ]

    for prompt in test_prompts:
        chat = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        tokens = tokenizer.encode(chat)
        input_ids = torch.tensor([tokens], device=device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                stop_tokens=tokenizer.get_stop_tokens(),
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        response = tokenizer.decode(output[0].tolist())
        # Extract assistant response
        if '<|im_start|>assistant\n' in response:
            response = response.split('<|im_start|>assistant\n')[-1]
        if '<|im_end|>' in response:
            response = response.split('<|im_end|>')[0]

        print(f"Q: {prompt}")
        print(f"A: {response.strip()}")
        print()


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimal 10M Model Training Pipeline"
    )
    parser.add_argument('--start-phase', type=int, default=1, choices=[1, 2, 3],
                        help='Phase to start from (1=pretrain, 2=distill, 3=finetune)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation on best available model')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BRANDON TINY - 10M OPTIMAL TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Device: {args.device}")
    print(f"  Output: {OUTPUT_DIR}")
    if not args.eval_only:
        print(f"  Starting from: Phase {args.start_phase}")
    print()

    # Eval-only mode: find best available checkpoint
    if args.eval_only:
        for phase in ['phase3_finetune', 'phase2_distill', 'phase1_pretrain']:
            best = OUTPUT_DIR / phase / 'best.pt'
            if best.exists():
                evaluate_model(args.device, best)
                return
        print("No checkpoints found. Run training first.")
        return

    # Track checkpoint through phases
    best_checkpoint = None
    start_time = time.time()

    # --- Phase 1: Foundation Pretrain ---
    if args.start_phase <= 1:
        best_checkpoint = run_phase1(args.device)
    else:
        best_checkpoint = OUTPUT_DIR / 'phase1_pretrain' / 'best.pt'
        if not best_checkpoint.exists():
            print(f"ERROR: Phase 1 checkpoint not found at {best_checkpoint}")
            print("Run with --start-phase 1 first.")
            return

    # --- Phase 2: Knowledge Distillation ---
    if args.start_phase <= 2:
        best_checkpoint = run_phase2(args.device, best_checkpoint)
    elif (OUTPUT_DIR / 'phase2_distill' / 'best.pt').exists():
        best_checkpoint = OUTPUT_DIR / 'phase2_distill' / 'best.pt'
    # else: keep phase 1 checkpoint

    # --- Phase 3: Instruction Finetune ---
    if args.start_phase <= 3:
        best_checkpoint = run_phase3(args.device, best_checkpoint)

    elapsed = time.time() - start_time

    # --- Final Evaluation ---
    evaluate_model(args.device, best_checkpoint)

    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Final model: {best_checkpoint}")
    print(f"  Total time: {elapsed/3600:.1f} hours")
    print("=" * 60)


if __name__ == '__main__':
    main()
