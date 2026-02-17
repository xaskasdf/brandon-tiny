"""
Serious Benchmark Suite for Brandon-Tiny 10M Optimal

Standardized evaluation using academic benchmarks suitable for ultra-small models.
Results are comparable with BabyLM Challenge and Super Tiny LMs papers.

Benchmarks:
1. BLiMP (Benchmark of Linguistic Minimal Pairs) - Grammar knowledge
2. HellaSwag (0-shot) - Commonsense NLI
3. ARC-Easy (0-shot) - Science reasoning
4. PIQA (0-shot) - Physical intuition
5. Winogrande (0-shot) - Coreference resolution
6. LAMBADA - Language modeling (last word prediction)
7. Wikitext-2 Perplexity - Standard perplexity benchmark

Usage:
    pip install datasets
    python scripts/benchmark_serious.py [--checkpoint PATH] [--tasks all]
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import TinyLlama
from src.tokenizer import Tokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER_PATH = 'data/tokenizer_8k.model'


def log_prob_of_sequence(model, token_ids, device):
    """Compute total log probability of a token sequence."""
    if len(token_ids) < 2:
        return 0.0
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(input_tensor)
    log_probs = F.log_softmax(logits, dim=-1)
    total = 0.0
    for i in range(1, len(token_ids)):
        total += log_probs[0, i - 1, token_ids[i]].item()
    return total


def log_prob_of_continuation(model, context_ids, continuation_ids, device):
    """Compute log probability of continuation given context."""
    full_ids = context_ids + continuation_ids
    if len(full_ids) > 512:
        full_ids = full_ids[-512:]
        # Recompute where continuation starts
        ctx_len = len(full_ids) - len(continuation_ids)
    else:
        ctx_len = len(context_ids)

    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(input_tensor)
    log_probs = F.log_softmax(logits, dim=-1)

    total = 0.0
    for i, token_id in enumerate(continuation_ids):
        pos = ctx_len + i - 1
        if pos >= 0 and pos < logits.shape[1]:
            total += log_probs[0, pos, token_id].item()
    return total


# =============================================================================
# 1. BLiMP - Benchmark of Linguistic Minimal Pairs
# =============================================================================

BLIMP_SUBTASKS = [
    'anaphor_gender_agreement',
    'anaphor_number_agreement',
    'determiner_noun_agreement_1',
    'determiner_noun_agreement_2',
    'determiner_noun_agreement_irregular_1',
    'determiner_noun_agreement_irregular_2',
    'determiner_noun_agreement_with_adj_1',
    'determiner_noun_agreement_with_adj_2',
    'determiner_noun_agreement_with_adj_irregular_1',
    'determiner_noun_agreement_with_adj_irregular_2',
    'existential_there_quantifiers_1',
    'existential_there_quantifiers_2',
    'irregular_past_participle_adjectives',
    'irregular_past_participle_verbs',
    'regular_plural_subject_verb_agreement_1',
    'regular_plural_subject_verb_agreement_2',
    'sentential_negation_npi_licensor_present',
    'sentential_negation_npi_scope',
    'wh_questions_object_gap',
    'wh_questions_subject_gap',
    'wh_questions_subject_gap_long_distance',
    'wh_vs_that_no_gap',
    'wh_vs_that_no_gap_long_distance',
    'wh_vs_that_with_gap',
    'wh_vs_that_with_gap_long_distance',
]


def eval_blimp(model, tokenizer, max_examples=None):
    """Evaluate on BLiMP - binary choice between grammatical/ungrammatical."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("  BLiMP - Benchmark of Linguistic Minimal Pairs")
    print("=" * 60)

    subtask_results = {}
    total_correct = 0
    total_count = 0

    for subtask in BLIMP_SUBTASKS:
        try:
            ds = load_dataset('blimp', subtask, split='train', trust_remote_code=True)
        except Exception as e:
            print(f"  [SKIP] {subtask}: {e}")
            continue

        correct = 0
        count = 0

        for example in ds:
            if max_examples and count >= max_examples:
                break

            good = example['sentence_good']
            bad = example['sentence_bad']

            good_tokens = tokenizer.encode(good)
            bad_tokens = tokenizer.encode(bad)

            good_score = log_prob_of_sequence(model, good_tokens, DEVICE)
            bad_score = log_prob_of_sequence(model, bad_tokens, DEVICE)

            if good_score > bad_score:
                correct += 1
            count += 1

        acc = correct / count if count > 0 else 0
        subtask_results[subtask] = {'accuracy': acc, 'correct': correct, 'total': count}
        total_correct += correct
        total_count += count

        print(f"  {subtask:50s}: {acc:.3f} ({correct}/{count})")

    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"\n  {'OVERALL BLiMP':50s}: {overall_acc:.3f} ({total_correct}/{total_count})")
    print(f"  Random baseline: 0.500")

    return {
        'overall_accuracy': overall_acc,
        'total_correct': total_correct,
        'total_count': total_count,
        'subtasks': subtask_results,
    }


# =============================================================================
# 2. HellaSwag - Commonsense NLI (4-way choice)
# =============================================================================

def eval_hellaswag(model, tokenizer, max_examples=1000):
    """Evaluate on HellaSwag - choose best continuation from 4 options."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("  HellaSwag - Commonsense NLI (0-shot)")
    print("=" * 60)

    ds = load_dataset('Rowan/hellaswag', split='validation', trust_remote_code=True)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        ctx = example['ctx']
        endings = example['endings']
        label = int(example['label'])

        ctx_tokens = tokenizer.encode(ctx)

        scores = []
        for ending in endings:
            end_tokens = tokenizer.encode(ending)
            score = log_prob_of_continuation(model, ctx_tokens, end_tokens, DEVICE)
            # Normalize by length
            score = score / max(len(end_tokens), 1)
            scores.append(score)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(ds)}, Acc: {correct/total:.3f}")

    acc = correct / total
    print(f"\n  HellaSwag Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"  Random baseline: 0.250 (4-way)")

    return {'accuracy': acc, 'correct': correct, 'total': total}


# =============================================================================
# 3. ARC-Easy - Science Reasoning
# =============================================================================

def eval_arc_easy(model, tokenizer, max_examples=1000):
    """Evaluate on ARC-Easy - multiple choice science questions."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("  ARC-Easy - Science Reasoning (0-shot)")
    print("=" * 60)

    ds = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='test', trust_remote_code=True)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        question = example['question']
        choices = example['choices']
        answer_key = example['answerKey']

        # Map answer key to index
        labels = choices['label']
        texts = choices['text']
        answer_idx = labels.index(answer_key) if answer_key in labels else -1

        if answer_idx == -1:
            continue

        ctx_tokens = tokenizer.encode(f"Question: {question}\nAnswer:")

        scores = []
        for choice_text in texts:
            choice_tokens = tokenizer.encode(f" {choice_text}")
            score = log_prob_of_continuation(model, ctx_tokens, choice_tokens, DEVICE)
            score = score / max(len(choice_tokens), 1)
            scores.append(score)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == answer_idx:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(ds)}, Acc: {correct/total:.3f}")

    acc = correct / total if total > 0 else 0
    n_choices = len(texts) if total > 0 else 4
    print(f"\n  ARC-Easy Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"  Random baseline: {1/n_choices:.3f} ({n_choices}-way)")

    return {'accuracy': acc, 'correct': correct, 'total': total}


# =============================================================================
# 4. PIQA - Physical Intuition QA
# =============================================================================

def eval_piqa(model, tokenizer, max_examples=1000):
    """Evaluate on PIQA - physical intuition QA (2-way choice)."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("  PIQA - Physical Intuition QA (0-shot)")
    print("=" * 60)

    ds = load_dataset('ybisk/piqa', split='validation', trust_remote_code=True)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        goal = example['goal']
        sol1 = example['sol1']
        sol2 = example['sol2']
        label = example['label']  # 0 or 1

        ctx_tokens = tokenizer.encode(f"Goal: {goal}\nSolution:")

        scores = []
        for sol in [sol1, sol2]:
            sol_tokens = tokenizer.encode(f" {sol}")
            score = log_prob_of_continuation(model, ctx_tokens, sol_tokens, DEVICE)
            score = score / max(len(sol_tokens), 1)
            scores.append(score)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(ds)}, Acc: {correct/total:.3f}")

    acc = correct / total
    print(f"\n  PIQA Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"  Random baseline: 0.500 (2-way)")

    return {'accuracy': acc, 'correct': correct, 'total': total}


# =============================================================================
# 5. Winogrande - Coreference Resolution
# =============================================================================

def eval_winogrande(model, tokenizer, max_examples=1000):
    """Evaluate on Winogrande - fill in the blank (2-way)."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("  Winogrande - Coreference Resolution (0-shot)")
    print("=" * 60)

    ds = load_dataset('allenai/winogrande', 'winogrande_xl', split='validation', trust_remote_code=True)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        sentence = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        answer = int(example['answer']) - 1  # 1-indexed to 0-indexed

        # Replace _ with each option
        sent1 = sentence.replace('_', option1)
        sent2 = sentence.replace('_', option2)

        score1 = log_prob_of_sequence(model, tokenizer.encode(sent1), DEVICE)
        score2 = log_prob_of_sequence(model, tokenizer.encode(sent2), DEVICE)

        # Normalize by length
        score1 /= max(len(tokenizer.encode(sent1)), 1)
        score2 /= max(len(tokenizer.encode(sent2)), 1)

        pred = 0 if score1 > score2 else 1
        if pred == answer:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(ds)}, Acc: {correct/total:.3f}")

    acc = correct / total
    print(f"\n  Winogrande Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"  Random baseline: 0.500 (2-way)")

    return {'accuracy': acc, 'correct': correct, 'total': total}


# =============================================================================
# 6. LAMBADA - Last Word Prediction
# =============================================================================

def eval_lambada(model, tokenizer, max_examples=1000):
    """Evaluate on LAMBADA - predict the last word of a passage."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("  LAMBADA - Last Word Prediction (0-shot)")
    print("=" * 60)

    ds = load_dataset('EleutherAI/lambada_openai', 'default', split='test', trust_remote_code=True)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    total = 0
    total_log_prob = 0.0
    total_words = 0

    for i, example in enumerate(ds):
        text = example['text']

        # Split into context + last word
        words = text.rsplit(' ', 1)
        if len(words) != 2:
            continue
        context, last_word = words

        ctx_tokens = tokenizer.encode(context)
        last_tokens = tokenizer.encode(f" {last_word}")

        if not last_tokens:
            continue

        # Get model's prediction
        full_tokens = ctx_tokens + last_tokens
        if len(full_tokens) > 512:
            full_tokens = full_tokens[-512:]
            ctx_len = len(full_tokens) - len(last_tokens)
        else:
            ctx_len = len(ctx_tokens)

        input_tensor = torch.tensor([full_tokens], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits, _ = model(input_tensor)

        # Check if greedy prediction matches first token of last word
        pred_pos = ctx_len - 1
        if pred_pos >= 0 and pred_pos < logits.shape[1]:
            greedy_pred = logits[0, pred_pos].argmax().item()
            if greedy_pred == last_tokens[0]:
                correct += 1

            # Log probability
            lp = F.log_softmax(logits[0, pred_pos], dim=-1)
            total_log_prob += lp[last_tokens[0]].item()
            total_words += 1

        total += 1

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(ds)}, Acc: {correct/total:.3f}")

    acc = correct / total if total > 0 else 0
    ppl = math.exp(-total_log_prob / total_words) if total_words > 0 else float('inf')

    print(f"\n  LAMBADA Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"  LAMBADA Perplexity: {ppl:.2f}")

    return {'accuracy': acc, 'perplexity': ppl, 'correct': correct, 'total': total}


# =============================================================================
# 7. Wikitext-2 Perplexity
# =============================================================================

def eval_wikitext_perplexity(model, tokenizer, max_tokens=100000):
    """Standard wikitext-2 perplexity."""
    print("\n" + "=" * 60)
    print("  Wikitext-2 Perplexity")
    print("=" * 60)

    test_path = Path('data/wikitext/test.txt')
    if not test_path.exists():
        print("  [SKIP] data/wikitext/test.txt not found")
        return None

    with open(test_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = tokenizer.encode(text)[:max_tokens]
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=DEVICE)

    total_loss = 0.0
    total_tokens = 0
    seq_len = 512

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

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"  Tokens evaluated: {total_tokens:,}")
    print(f"  Perplexity: {ppl:.2f}")

    return {'perplexity': ppl, 'avg_loss': avg_loss, 'tokens': total_tokens}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Serious Benchmarks for Brandon-Tiny')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/10m_optimal/phase3_finetune/best.pt')
    parser.add_argument('--name', type=str, default='Brandon-Tiny-10M-Optimal')
    parser.add_argument('--tasks', type=str, default='all',
                        help='Comma-separated: blimp,hellaswag,arc,piqa,winogrande,lambada,perplexity')
    parser.add_argument('--max-examples', type=int, default=1000,
                        help='Max examples per task (0=all)')
    parser.add_argument('--output', type=str, default='checkpoints/serious_benchmark.json')
    args = parser.parse_args()

    max_ex = args.max_examples if args.max_examples > 0 else None
    tasks = args.tasks.split(',') if args.tasks != 'all' else [
        'blimp', 'hellaswag', 'arc', 'piqa', 'winogrande', 'lambada', 'perplexity'
    ]

    print(f"\n{'=' * 60}")
    print(f"  BRANDON-TINY SERIOUS BENCHMARKS")
    print(f"  Model: {args.name}")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"  Max examples: {max_ex or 'all'}")
    print(f"{'=' * 60}")

    model = TinyLlama.from_checkpoint(args.checkpoint, device=DEVICE)
    model.eval()
    tokenizer = Tokenizer(TOKENIZER_PATH)

    print(f"  Parameters: {model.count_parameters():,}")

    results = {
        'model_name': args.name,
        'checkpoint': args.checkpoint,
        'parameters': model.count_parameters(),
    }

    if 'perplexity' in tasks:
        results['wikitext2_perplexity'] = eval_wikitext_perplexity(model, tokenizer)

    if 'blimp' in tasks:
        results['blimp'] = eval_blimp(model, tokenizer, max_examples=max_ex)

    if 'hellaswag' in tasks:
        results['hellaswag'] = eval_hellaswag(model, tokenizer, max_examples=max_ex)

    if 'arc' in tasks:
        results['arc_easy'] = eval_arc_easy(model, tokenizer, max_examples=max_ex)

    if 'piqa' in tasks:
        results['piqa'] = eval_piqa(model, tokenizer, max_examples=max_ex)

    if 'winogrande' in tasks:
        results['winogrande'] = eval_winogrande(model, tokenizer, max_examples=max_ex)

    if 'lambada' in tasks:
        results['lambada'] = eval_lambada(model, tokenizer, max_examples=max_ex)

    # Final summary
    print(f"\n\n{'=' * 60}")
    print(f"  FINAL RESULTS: {args.name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"{'=' * 60}")

    print(f"\n  {'Benchmark':25s} | {'Score':>8s} | {'Random':>8s} | {'Delta':>8s}")
    print(f"  {'─' * 25}-+-{'─' * 8}-+-{'─' * 8}-+-{'─' * 8}")

    comparisons = {
        'blimp': ('overall_accuracy', 0.500),
        'hellaswag': ('accuracy', 0.250),
        'arc_easy': ('accuracy', 0.250),
        'piqa': ('accuracy', 0.500),
        'winogrande': ('accuracy', 0.500),
        'lambada': ('accuracy', 0.000),
    }

    for task_name, (metric, random_baseline) in comparisons.items():
        if task_name in results and results[task_name]:
            score = results[task_name][metric]
            delta = score - random_baseline
            sign = '+' if delta >= 0 else ''
            print(f"  {task_name:25s} | {score:>7.3f} | {random_baseline:>7.3f} | {sign}{delta:>7.3f}")

    if 'wikitext2_perplexity' in results and results['wikitext2_perplexity']:
        ppl = results['wikitext2_perplexity']['perplexity']
        print(f"  {'wikitext2_ppl':25s} | {ppl:>8.1f} |")

    if 'lambada' in results and results['lambada']:
        lppl = results['lambada']['perplexity']
        print(f"  {'lambada_ppl':25s} | {lppl:>8.1f} |")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"\n  Results saved to: {output_path}")
    print(f"  Ready for HuggingFace model card!")


if __name__ == '__main__':
    main()
