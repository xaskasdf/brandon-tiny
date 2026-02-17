"""
Comprehensive Benchmark Suite for Brandon-Tiny 10M Optimal Model

Evaluates:
1. Perplexity on wikitext-2 test set (standard corpus)
2. Generation quality scoring (automated + samples for human review)
3. Repetition analysis (n-gram repetition rates in free generation)
4. Instruction following rate (does the model actually answer the question?)

Usage:
    python scripts/benchmark_10m.py [--checkpoint PATH] [--all-models]
"""

import sys
import json
import time
import math
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import TinyLlama, ModelConfig
from src.tokenizer import Tokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER_PATH = 'data/tokenizer_8k.model'


# =============================================================================
# TEST PROMPTS for instruction following & generation quality
# =============================================================================

INSTRUCTION_PROMPTS = [
    # Simple factual
    {"prompt": "What is the capital of France?", "keywords": ["paris"], "category": "factual"},
    {"prompt": "What color is the sky?", "keywords": ["blue"], "category": "factual"},
    {"prompt": "How many legs does a dog have?", "keywords": ["four", "4"], "category": "factual"},
    {"prompt": "What is water made of?", "keywords": ["hydrogen", "oxygen", "h2o", "water"], "category": "factual"},
    {"prompt": "What planet do we live on?", "keywords": ["earth"], "category": "factual"},

    # Simple math
    {"prompt": "What is 2 + 2?", "keywords": ["4", "four"], "category": "math"},
    {"prompt": "What is 10 minus 3?", "keywords": ["7", "seven"], "category": "math"},
    {"prompt": "What is 5 times 2?", "keywords": ["10", "ten"], "category": "math"},

    # Explanation/reasoning
    {"prompt": "Why is the sky blue?", "keywords": ["light", "scatter", "sun", "atmosphere"], "category": "reasoning"},
    {"prompt": "Why do we need to sleep?", "keywords": ["rest", "body", "brain", "energy", "health"], "category": "reasoning"},
    {"prompt": "What happens when ice melts?", "keywords": ["water", "liquid", "warm", "heat"], "category": "reasoning"},
    {"prompt": "Why do birds fly?", "keywords": ["wing", "air", "fly", "feather"], "category": "reasoning"},

    # Creative/open-ended
    {"prompt": "Tell me a short story about a cat.", "keywords": ["cat", "the"], "category": "creative"},
    {"prompt": "Write a poem about the moon.", "keywords": ["moon", "night", "light", "sky"], "category": "creative"},
    {"prompt": "Describe a beautiful sunset.", "keywords": ["sun", "sky", "color", "red", "orange", "light"], "category": "creative"},

    # Identity
    {"prompt": "What is your name?", "keywords": ["brandon", "tiny", "assistant", "ai", "language"], "category": "identity"},
    {"prompt": "Who are you?", "keywords": ["brandon", "tiny", "assistant", "ai", "model", "language"], "category": "identity"},

    # Instructions
    {"prompt": "List three fruits.", "keywords": ["apple", "banana", "orange", "grape", "mango", "berry", "pear", "peach", "melon", "cherry", "lemon", "kiwi", "plum", "fig"], "category": "list"},
    {"prompt": "Name two colors.", "keywords": ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown"], "category": "list"},
    {"prompt": "Give me an example of a mammal.", "keywords": ["dog", "cat", "whale", "elephant", "human", "bear", "lion", "horse", "cow", "deer", "mouse", "rabbit", "monkey", "tiger"], "category": "list"},

    # Continuation / completion
    {"prompt": "The sun rises in the", "keywords": ["east", "morning", "sky"], "category": "completion"},
    {"prompt": "Water freezes at", "keywords": ["0", "32", "zero", "degree", "celsius", "fahrenheit"], "category": "completion"},

    # Conversation
    {"prompt": "Hello! How are you today?", "keywords": ["hello", "hi", "good", "well", "great", "fine", "thank", "doing"], "category": "conversation"},
    {"prompt": "Thank you for your help!", "keywords": ["welcome", "glad", "happy", "help", "thank", "pleasure"], "category": "conversation"},
    {"prompt": "Can you help me with something?", "keywords": ["yes", "sure", "of course", "help", "happy", "glad", "what"], "category": "conversation"},
]

# Free generation prompts (for repetition analysis)
FREE_GENERATION_PROMPTS = [
    "Once upon a time, there was a",
    "The most important thing about science is",
    "In a small village by the sea,",
    "The future of technology will",
    "A good friend is someone who",
    "Education is important because",
    "The forest was dark and mysterious.",
    "When I think about happiness,",
    "The old man walked slowly down the",
    "Music has the power to",
    "The little girl found a magic",
    "In the year 2050,",
    "The secret to a good life is",
    "Rain began to fall as the",
    "The scientist discovered that",
    "A wise teacher once said,",
    "The mountain was covered in",
    "Every morning, the birds would",
    "The recipe for success includes",
    "Deep in the ocean, there lives a",
]


# =============================================================================
# 1. PERPLEXITY on Wikitext-2
# =============================================================================

def compute_perplexity(model, tokenizer, text_path, max_tokens=100000, seq_len=512):
    """Compute perplexity on a standard text corpus."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 1: Perplexity on Wikitext-2 Test")
    print("=" * 60)

    # Read and tokenize
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    print(f"  Corpus: {text_path}")
    print(f"  Tokens: {len(tokens):,} (max {max_tokens:,})")
    print(f"  Sequence length: {seq_len}")

    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=DEVICE)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_chunks = 0

    # Process in non-overlapping chunks
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(tokens) - seq_len, seq_len):
            chunk = tokens_tensor[i:i + seq_len + 1].unsqueeze(0)
            input_ids = chunk[:, :-1]
            targets = chunk[:, 1:]

            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
            n_chunks += 1

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"\n  Results:")
    print(f"    Chunks processed: {n_chunks}")
    print(f"    Tokens evaluated: {total_tokens:,}")
    print(f"    Average NLL loss: {avg_loss:.4f}")
    print(f"    Perplexity: {perplexity:.2f}")

    # Context: what perplexity means at different levels
    print(f"\n  Context (approximate):")
    print(f"    PPL < 20:   Very good (GPT-2 level on wikitext-2)")
    print(f"    PPL 20-50:  Good (small fine-tuned models)")
    print(f"    PPL 50-200: Reasonable for tiny models (<50M)")
    print(f"    PPL > 200:  Model struggles with this domain")

    return {"avg_loss": avg_loss, "perplexity": perplexity, "tokens_evaluated": total_tokens}


# =============================================================================
# 2. GENERATION QUALITY (automated scoring + samples)
# =============================================================================

def generate_response(model, tokenizer, prompt, max_tokens=150, temperature=0.7,
                      top_p=0.9, top_k=50, repetition_penalty=1.2):
    """Generate a response using ChatML format."""
    chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_tokens = tokenizer.encode(chat_prompt)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=DEVICE)

    stop_tokens = tokenizer.get_stop_tokens()

    output = model.generate(
        input_tensor,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_tokens=stop_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=3,
    )

    # Decode only the generated part
    generated_tokens = output[0, len(input_tokens):].tolist()

    # Remove stop tokens from end
    stop_set = set(stop_tokens)
    while generated_tokens and generated_tokens[-1] in stop_set:
        generated_tokens.pop()

    response = tokenizer.decode(generated_tokens)
    return response.strip(), generated_tokens


def score_generation_quality(response, prompt_info):
    """Score a single generation on multiple dimensions."""
    scores = {}
    response_lower = response.lower()

    # 1. Non-empty response
    scores['non_empty'] = 1.0 if len(response.strip()) > 0 else 0.0

    # 2. Minimum length (at least 5 words)
    word_count = len(response.split())
    scores['min_length'] = 1.0 if word_count >= 3 else (word_count / 3.0)

    # 3. Not just repeating the prompt
    prompt_lower = prompt_info['prompt'].lower()
    prompt_words = set(prompt_lower.split())
    response_words = set(response_lower.split())
    new_words = response_words - prompt_words
    scores['novelty'] = min(1.0, len(new_words) / max(1, len(response_words)))

    # 4. Keyword match (instruction following)
    if prompt_info['keywords']:
        matched = sum(1 for kw in prompt_info['keywords'] if kw in response_lower)
        scores['keyword_match'] = min(1.0, matched / 1.0)  # At least 1 keyword
    else:
        scores['keyword_match'] = 1.0

    # 5. Coherence: no excessive special chars or garbage
    alpha_ratio = sum(1 for c in response if c.isalpha()) / max(1, len(response))
    scores['coherence'] = 1.0 if alpha_ratio > 0.5 else alpha_ratio / 0.5

    # 6. No excessive repetition within response
    words = response_lower.split()
    if len(words) > 3:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        max_repeat = max(bigram_counts.values()) if bigram_counts else 0
        total_bigrams = len(bigrams)
        repeat_ratio = max_repeat / total_bigrams
        scores['low_repetition'] = 1.0 if repeat_ratio < 0.15 else max(0, 1.0 - (repeat_ratio - 0.15) * 5)
    else:
        scores['low_repetition'] = 1.0

    # 7. Grammaticality heuristic (starts with capital, has periods)
    has_structure = (
        (response[0].isupper() if response else False) or
        ('.' in response) or
        ('!' in response) or
        (',' in response)
    )
    scores['structure'] = 1.0 if has_structure else 0.5

    # Overall score (weighted)
    weights = {
        'non_empty': 0.15,
        'min_length': 0.10,
        'novelty': 0.10,
        'keyword_match': 0.25,
        'coherence': 0.15,
        'low_repetition': 0.15,
        'structure': 0.10,
    }
    overall = sum(scores[k] * weights[k] for k in weights)
    scores['overall'] = overall

    return scores


def benchmark_generation_quality(model, tokenizer):
    """Run generation quality benchmark on all prompts."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 2: Generation Quality (Automated Scoring)")
    print("=" * 60)

    results = []
    category_scores = defaultdict(list)

    for i, pinfo in enumerate(INSTRUCTION_PROMPTS):
        prompt = pinfo['prompt']
        response, tokens = generate_response(model, tokenizer, prompt)
        scores = score_generation_quality(response, pinfo)

        category_scores[pinfo['category']].append(scores['overall'])

        result = {
            'prompt': prompt,
            'response': response,
            'category': pinfo['category'],
            'scores': scores,
            'token_count': len(tokens),
        }
        results.append(result)

        # Print each result
        status = "PASS" if scores['overall'] >= 0.5 else "FAIL"
        print(f"\n  [{i+1:2d}/{len(INSTRUCTION_PROMPTS)}] [{status}] ({scores['overall']:.2f}) {pinfo['category']}")
        print(f"      Q: {prompt}")
        resp_preview = response[:120] + "..." if len(response) > 120 else response
        print(f"      A: {resp_preview}")

        # Show subscores for failures
        if scores['overall'] < 0.5:
            low_scores = {k: v for k, v in scores.items() if v < 0.5 and k != 'overall'}
            if low_scores:
                print(f"      Low scores: {low_scores}")

    # Summary
    all_scores = [r['scores']['overall'] for r in results]
    avg_score = sum(all_scores) / len(all_scores)
    pass_rate = sum(1 for s in all_scores if s >= 0.5) / len(all_scores)

    print(f"\n  {'─' * 50}")
    print(f"  Overall Quality Score: {avg_score:.3f} / 1.000")
    print(f"  Pass Rate (>=0.5): {pass_rate*100:.1f}% ({sum(1 for s in all_scores if s >= 0.5)}/{len(all_scores)})")

    print(f"\n  By Category:")
    for cat, cat_scores in sorted(category_scores.items()):
        cat_avg = sum(cat_scores) / len(cat_scores)
        cat_pass = sum(1 for s in cat_scores if s >= 0.5) / len(cat_scores)
        print(f"    {cat:15s}: avg={cat_avg:.3f}, pass={cat_pass*100:.0f}% ({len(cat_scores)} prompts)")

    return {
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "category_scores": {k: sum(v)/len(v) for k, v in category_scores.items()},
        "results": results,
    }


# =============================================================================
# 3. REPETITION ANALYSIS
# =============================================================================

def compute_ngram_repetition(tokens, n):
    """Compute the fraction of n-grams that are repeated."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    repeated_fraction = 1.0 - (unique / total)
    return repeated_fraction


def compute_sequence_repetition(text):
    """Detect if the model gets stuck in a loop (repeating phrases)."""
    words = text.lower().split()
    if len(words) < 10:
        return 0.0

    # Check for repeating subsequences of length 3-10
    max_repeat_score = 0.0
    for subseq_len in range(3, min(11, len(words) // 2)):
        for start in range(len(words) - subseq_len * 2):
            pattern = words[start:start + subseq_len]
            # Count consecutive repetitions
            reps = 1
            pos = start + subseq_len
            while pos + subseq_len <= len(words):
                if words[pos:pos + subseq_len] == pattern:
                    reps += 1
                    pos += subseq_len
                else:
                    break
            if reps >= 2:
                score = (reps * subseq_len) / len(words)
                max_repeat_score = max(max_repeat_score, score)

    return max_repeat_score


def benchmark_repetition(model, tokenizer):
    """Analyze repetition rates in free generation."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 3: Repetition Analysis (Free Generation)")
    print("=" * 60)

    all_bigram_reps = []
    all_trigram_reps = []
    all_4gram_reps = []
    all_seq_reps = []
    all_responses = []

    for i, prompt in enumerate(FREE_GENERATION_PROMPTS):
        response, gen_tokens = generate_response(
            model, tokenizer, prompt,
            max_tokens=200, temperature=0.7,
            repetition_penalty=1.2
        )

        bigram_rep = compute_ngram_repetition(gen_tokens, 2)
        trigram_rep = compute_ngram_repetition(gen_tokens, 3)
        fourgram_rep = compute_ngram_repetition(gen_tokens, 4)
        seq_rep = compute_sequence_repetition(response)

        all_bigram_reps.append(bigram_rep)
        all_trigram_reps.append(trigram_rep)
        all_4gram_reps.append(fourgram_rep)
        all_seq_reps.append(seq_rep)

        all_responses.append({
            'prompt': prompt,
            'response': response,
            'token_count': len(gen_tokens),
            'bigram_rep': bigram_rep,
            'trigram_rep': trigram_rep,
            'seq_rep': seq_rep,
        })

        status = "OK" if seq_rep < 0.3 else "REP!"
        print(f"  [{i+1:2d}/{len(FREE_GENERATION_PROMPTS)}] [{status}] bi={bigram_rep:.2f} tri={trigram_rep:.2f} seq={seq_rep:.2f} | {len(gen_tokens)} tok")
        resp_preview = response[:100] + "..." if len(response) > 100 else response
        print(f"      {resp_preview}")

    # Summary
    avg_bigram = sum(all_bigram_reps) / len(all_bigram_reps)
    avg_trigram = sum(all_trigram_reps) / len(all_trigram_reps)
    avg_4gram = sum(all_4gram_reps) / len(all_4gram_reps)
    avg_seq = sum(all_seq_reps) / len(all_seq_reps)
    stuck_count = sum(1 for s in all_seq_reps if s >= 0.3)

    print(f"\n  {'─' * 50}")
    print(f"  Average Repetition Rates:")
    print(f"    Bigram:  {avg_bigram:.3f} (lower = more diverse)")
    print(f"    Trigram: {avg_trigram:.3f}")
    print(f"    4-gram:  {avg_4gram:.3f}")
    print(f"    Sequence loop: {avg_seq:.3f}")
    print(f"  Stuck in loop (seq>0.3): {stuck_count}/{len(FREE_GENERATION_PROMPTS)}")

    print(f"\n  Reference ranges (human text):")
    print(f"    Bigram rep:  0.30-0.50 (some repetition is natural)")
    print(f"    Trigram rep: 0.05-0.15")
    print(f"    4-gram rep:  0.01-0.05")
    print(f"    Seq loop:    <0.05 (no stuck loops)")

    return {
        "avg_bigram_rep": avg_bigram,
        "avg_trigram_rep": avg_trigram,
        "avg_4gram_rep": avg_4gram,
        "avg_seq_rep": avg_seq,
        "stuck_count": stuck_count,
        "responses": all_responses,
    }


# =============================================================================
# 4. INSTRUCTION FOLLOWING RATE
# =============================================================================

def benchmark_instruction_following(generation_results):
    """Analyze instruction following from generation benchmark results."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 4: Instruction Following Analysis")
    print("=" * 60)

    category_stats = defaultdict(lambda: {'total': 0, 'follow': 0, 'partial': 0, 'fail': 0})

    for r in generation_results['results']:
        cat = r['category']
        score = r['scores']['keyword_match']
        category_stats[cat]['total'] += 1

        if score >= 0.8:
            category_stats[cat]['follow'] += 1
        elif score >= 0.3:
            category_stats[cat]['partial'] += 1
        else:
            category_stats[cat]['fail'] += 1

    total_follow = sum(s['follow'] for s in category_stats.values())
    total_partial = sum(s['partial'] for s in category_stats.values())
    total_fail = sum(s['fail'] for s in category_stats.values())
    total = sum(s['total'] for s in category_stats.values())

    print(f"\n  Instruction Following by Category:")
    print(f"  {'Category':15s} | {'Follow':>7s} | {'Partial':>7s} | {'Fail':>7s} | {'Rate':>7s}")
    print(f"  {'─'*15}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}")

    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        rate = (s['follow'] + s['partial'] * 0.5) / s['total']
        print(f"  {cat:15s} | {s['follow']:>7d} | {s['partial']:>7d} | {s['fail']:>7d} | {rate:>6.1%}")

    overall_rate = (total_follow + total_partial * 0.5) / total
    print(f"  {'─'*15}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}")
    print(f"  {'TOTAL':15s} | {total_follow:>7d} | {total_partial:>7d} | {total_fail:>7d} | {overall_rate:>6.1%}")

    # Detailed failures
    failures = [r for r in generation_results['results'] if r['scores']['keyword_match'] < 0.3]
    if failures:
        print(f"\n  Failed Prompts ({len(failures)}):")
        for r in failures:
            print(f"    Q: {r['prompt']}")
            resp_preview = r['response'][:80] + "..." if len(r['response']) > 80 else r['response']
            print(f"    A: {resp_preview}")
            print()

    return {
        "overall_rate": overall_rate,
        "follow_count": total_follow,
        "partial_count": total_partial,
        "fail_count": total_fail,
        "total": total,
        "by_category": {k: dict(v) for k, v in category_stats.items()},
    }


# =============================================================================
# 5. SPEED BENCHMARK
# =============================================================================

def benchmark_speed(model, tokenizer):
    """Measure inference speed (tokens/second)."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 5: Inference Speed")
    print("=" * 60)

    prompt = "The quick brown fox"
    input_tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=DEVICE)

    # Warmup
    for _ in range(3):
        model.generate(input_tensor, max_new_tokens=20, temperature=0.8)

    # Benchmark
    gen_lengths = [32, 64, 128, 256]
    for n_tokens in gen_lengths:
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        start = time.perf_counter()

        n_runs = 5
        total_generated = 0
        for _ in range(n_runs):
            output = model.generate(input_tensor, max_new_tokens=n_tokens, temperature=0.8)
            total_generated += output.shape[1] - len(input_tokens)

        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        elapsed = time.perf_counter() - start

        tok_per_sec = total_generated / elapsed
        print(f"  {n_tokens:3d} tokens: {tok_per_sec:.1f} tok/s ({elapsed/n_runs*1000:.0f}ms/run)")

    # Memory usage
    if DEVICE == 'cuda':
        mem_allocated = torch.cuda.memory_allocated() / 1e6
        mem_reserved = torch.cuda.memory_reserved() / 1e6
        print(f"\n  GPU Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved")

    # Model size
    param_count = model.count_parameters()
    model_size_mb = param_count * 4 / 1e6  # float32
    model_size_bf16 = param_count * 2 / 1e6  # bfloat16
    print(f"  Model size: {model_size_mb:.1f}MB (fp32), {model_size_bf16:.1f}MB (bf16)")
    print(f"  Parameters: {param_count:,}")

    return {"param_count": param_count}


# =============================================================================
# MAIN: Run all benchmarks
# =============================================================================

def run_all_benchmarks(checkpoint_path, model_name="10M Optimal"):
    """Run the complete benchmark suite on a model."""
    print("\n" + "=" * 60)
    print(f"  BRANDON-TINY BENCHMARK SUITE")
    print(f"  Model: {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Load model
    print(f"\n  Loading model...")
    model = TinyLlama.from_checkpoint(checkpoint_path, device=DEVICE)
    model.eval()
    tokenizer = Tokenizer(TOKENIZER_PATH)

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Config: dim={model.config.dim}, layers={model.config.n_layers}, "
          f"heads={model.config.n_heads}, sharing={model.config.block_sharing}")

    results = {"model_name": model_name, "checkpoint": checkpoint_path}

    # 1. Perplexity
    wikitext_path = Path('data/wikitext/test.txt')
    if wikitext_path.exists():
        results['perplexity'] = compute_perplexity(model, tokenizer, wikitext_path)
    else:
        print(f"\n  [SKIP] Wikitext test set not found at {wikitext_path}")

    # 2. Generation Quality
    results['generation'] = benchmark_generation_quality(model, tokenizer)

    # 3. Repetition
    results['repetition'] = benchmark_repetition(model, tokenizer)

    # 4. Instruction Following (uses generation results)
    results['instruction_following'] = benchmark_instruction_following(results['generation'])

    # 5. Speed
    results['speed'] = benchmark_speed(model, tokenizer)

    # =================================================================
    # FINAL REPORT CARD
    # =================================================================
    print("\n\n" + "=" * 60)
    print("  REPORT CARD: " + model_name)
    print("=" * 60)

    ppl = results.get('perplexity', {}).get('perplexity', float('nan'))
    gen_score = results['generation']['avg_score']
    gen_pass = results['generation']['pass_rate']
    rep_seq = results['repetition']['avg_seq_rep']
    rep_stuck = results['repetition']['stuck_count']
    instr_rate = results['instruction_following']['overall_rate']
    params = results['speed']['param_count']

    def grade(score, thresholds):
        """A/B/C/D/F grading."""
        for threshold, letter in thresholds:
            if score >= threshold:
                return letter
        return 'F'

    ppl_grade = grade(1.0/ppl * 100, [(2.0, 'A'), (1.0, 'B'), (0.5, 'C'), (0.2, 'D')]) if ppl < 10000 else 'F'
    gen_grade = grade(gen_score, [(0.8, 'A'), (0.65, 'B'), (0.5, 'C'), (0.35, 'D')])
    rep_grade = grade(1.0 - rep_seq, [(0.95, 'A'), (0.85, 'B'), (0.7, 'C'), (0.5, 'D')])
    instr_grade = grade(instr_rate, [(0.8, 'A'), (0.65, 'B'), (0.5, 'C'), (0.35, 'D')])

    print(f"\n  {'Metric':30s} | {'Value':>12s} | {'Grade':>5s}")
    print(f"  {'─'*30}-+-{'─'*12}-+-{'─'*5}")
    print(f"  {'Perplexity (wikitext-2)':30s} | {ppl:>12.2f} | {ppl_grade:>5s}")
    print(f"  {'Generation Quality':30s} | {gen_score:>12.3f} | {gen_grade:>5s}")
    print(f"  {'Generation Pass Rate':30s} | {gen_pass*100:>11.1f}% |")
    print(f"  {'Repetition (seq loop)':30s} | {rep_seq:>12.3f} | {rep_grade:>5s}")
    print(f"  {'Stuck in loop':30s} | {rep_stuck:>9d}/20 |")
    print(f"  {'Instruction Following':30s} | {instr_rate*100:>11.1f}% | {instr_grade:>5s}")
    print(f"  {'Parameters':30s} | {params:>12,} |")

    # Overall verdict
    grades = [ppl_grade, gen_grade, rep_grade, instr_grade]
    grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    avg_gpa = sum(grade_values.get(g, 0) for g in grades) / len(grades)
    overall_letter = grade(avg_gpa, [(3.5, 'A'), (2.5, 'B'), (1.5, 'C'), (0.5, 'D')])

    print(f"\n  OVERALL GRADE: {overall_letter} (GPA: {avg_gpa:.1f}/4.0)")
    print(f"  For a {params:,} parameter model running on palitos de helado")
    print("=" * 60)

    # Save full results
    output_path = Path('checkpoints') / 'benchmark_results.json'

    # Convert for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to: {output_path}")

    return results


def compare_models(checkpoints):
    """Run benchmarks on multiple models and compare."""
    all_results = []

    for name, path in checkpoints:
        if Path(path).exists():
            results = run_all_benchmarks(path, model_name=name)
            all_results.append(results)
        else:
            print(f"\n  [SKIP] {name}: checkpoint not found at {path}")

    if len(all_results) > 1:
        print("\n\n" + "=" * 80)
        print("  COMPARISON TABLE")
        print("=" * 80)

        header = f"  {'Model':25s} | {'PPL':>8s} | {'GenQ':>6s} | {'Pass%':>6s} | {'RepSeq':>6s} | {'Instr%':>6s}"
        print(header)
        print(f"  {'─'*25}-+-{'─'*8}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}")

        for r in all_results:
            name = r['model_name'][:25]
            ppl = r.get('perplexity', {}).get('perplexity', float('nan'))
            gen = r['generation']['avg_score']
            pass_r = r['generation']['pass_rate'] * 100
            rep = r['repetition']['avg_seq_rep']
            instr = r['instruction_following']['overall_rate'] * 100
            print(f"  {name:25s} | {ppl:>8.1f} | {gen:>6.3f} | {pass_r:>5.1f}% | {rep:>6.3f} | {instr:>5.1f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Brandon-Tiny Benchmark Suite')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/10m_optimal/phase3_finetune/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--name', type=str, default='10M Optimal',
                        help='Model name for reports')
    parser.add_argument('--all-models', action='store_true',
                        help='Compare all available models')
    args = parser.parse_args()

    if args.all_models:
        checkpoints = [
            ("10M Optimal (3-phase)", "checkpoints/10m_optimal/phase3_finetune/best.pt"),
            ("10M Enhanced v2", "checkpoints/10m_enhanced_v2/finetune/best.pt"),
            ("10M Dream", "checkpoints/10m_dream/finetune/best.pt"),
            ("10M Synthetic-only", "checkpoints/10m_synthetic_only/finetune/best.pt"),
            ("30M v2 Original", "checkpoints/30m_v2/finetune/best.pt"),
            ("30M v2 Wiki", "checkpoints/30m_v2_wiki/finetune/best.pt"),
            ("30M Dream", "checkpoints/30m_dream/finetune/best.pt"),
        ]
        compare_models(checkpoints)
    else:
        run_all_benchmarks(args.checkpoint, model_name=args.name)


if __name__ == '__main__':
    main()
