#!/usr/bin/env python
"""
Generate high-quality synthetic instruction data using a local LLM API.

Generates 4 types of data:
  1. Single-turn QA (English)
  2. Multi-turn conversations (English)
  3. Chain-of-thought reasoning (English)
  4. Spanish language examples (single-turn + CoT)

Usage:
    python scripts/generate_synthetic_llm.py --output data/chat/synthetic_10k.jsonl --num 10000
    python scripts/generate_synthetic_llm.py --output data/chat/val_synth.jsonl --num 1000
    python scripts/generate_synthetic_llm.py --test  # test API connection
"""

import argparse
import json
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

API_BASE = "http://localhost:5282"


def call_completion(
    prompt: str,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
    json_schema: Optional[dict] = None,
    temperature: float = 0.8,
    max_tokens: int = 512
) -> dict:
    """Call the /completion endpoint."""
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if json_mode:
        payload["json"] = True
    if json_schema:
        payload["json_schema"] = json_schema

    response = requests.post(f"{API_BASE}/completion", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


# ── Categories ──────────────────────────────────────────────────────

CATEGORIES = [
    "math", "science", "history", "geography", "language",
    "programming", "logic", "general_knowledge", "creative", "explanation",
]

TOPICS = {
    "math": ["addition", "subtraction", "multiplication", "fractions", "percentages", "geometry", "algebra"],
    "science": ["physics", "chemistry", "biology", "astronomy", "weather", "ecology", "geology"],
    "history": ["wars", "inventions", "empires", "revolutions", "discoveries", "ancient civilizations"],
    "geography": ["mountains", "rivers", "deserts", "islands", "climate", "countries", "capitals"],
    "programming": ["Python", "JavaScript", "algorithms", "data structures", "web", "databases", "git"],
    "general_knowledge": ["food", "sports", "music", "art", "technology", "health", "animals"],
    "explanation": ["technology", "nature", "society", "health", "economics", "philosophy"],
    "language": ["grammar", "vocabulary", "etymology", "idioms", "writing"],
    "logic": ["puzzles", "sequences", "riddles", "probability", "sets"],
    "creative": ["stories", "poetry", "descriptions", "brainstorming", "worldbuilding"],
}

# ── JSON Schemas ────────────────────────────────────────────────────

QA_SCHEMA = {
    "name": "qa_pair",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "A clear question or instruction from a user"
            },
            "response": {
                "type": "string",
                "description": "A helpful, accurate, and concise answer"
            }
        },
        "required": ["instruction", "response"],
        "additionalProperties": False
    }
}

MULTI_TURN_SCHEMA = {
    "name": "multi_turn_conversation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "user": {"type": "string"},
                        "assistant": {"type": "string"}
                    },
                    "required": ["user", "assistant"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["turns"],
        "additionalProperties": False
    }
}

COT_SCHEMA = {
    "name": "cot_reasoning",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "A question that requires step-by-step reasoning"
            },
            "thinking": {
                "type": "string",
                "description": "Step-by-step reasoning process to arrive at the answer"
            },
            "answer": {
                "type": "string",
                "description": "The final concise answer"
            }
        },
        "required": ["instruction", "thinking", "answer"],
        "additionalProperties": False
    }
}

SPANISH_QA_SCHEMA = {
    "name": "qa_spanish",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "A clear question in Spanish"
            },
            "response": {
                "type": "string",
                "description": "A helpful and accurate answer in Spanish"
            }
        },
        "required": ["instruction", "response"],
        "additionalProperties": False
    }
}

SPANISH_COT_SCHEMA = {
    "name": "cot_spanish",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "A question requiring step-by-step reasoning, in Spanish"
            },
            "thinking": {
                "type": "string",
                "description": "Step-by-step reasoning process in Spanish"
            },
            "answer": {
                "type": "string",
                "description": "The final concise answer in Spanish"
            }
        },
        "required": ["instruction", "thinking", "answer"],
        "additionalProperties": False
    }
}


# ── Generators ──────────────────────────────────────────────────────

def generate_single_turn(category: str, retry_count: int = 3) -> Optional[dict]:
    """Generate a single-turn Q&A pair."""
    topic = random.choice(TOPICS.get(category, ["general"]))

    system_prompt = (
        "You are an expert at creating high-quality educational Q&A pairs. "
        "Generate a realistic question and a clear, helpful answer. "
        "Keep answers concise (2-4 sentences). Vary style and difficulty."
    )

    prompt = f"""Generate a Q&A pair about {category} (topic: {topic}).

The question should sound natural, like a real person asking.
The answer should be accurate, educational, and concise (2-4 sentences)."""

    for attempt in range(retry_count):
        try:
            result = call_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_schema=QA_SCHEMA,
                temperature=0.9,
                max_tokens=300
            )
            resp = result.get("response", {})
            if isinstance(resp, dict) and resp.get("instruction") and resp.get("response"):
                instr = resp["instruction"].strip()
                ans = resp["response"].strip()
                if len(instr) > 10 and len(ans) > 20:
                    return {"instruction": instr, "input": "", "output": ans}
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(1)
            else:
                print(f"  [single-turn] Failed ({category}): {e}")
    return None


def generate_multi_turn(category: str, retry_count: int = 3) -> Optional[dict]:
    """Generate a multi-turn conversation (2-4 exchanges)."""
    topic = random.choice(TOPICS.get(category, ["general"]))
    n_turns = random.choice([2, 3, 4])

    system_prompt = (
        "You create realistic multi-turn conversations between a curious user and a helpful assistant. "
        "Each turn should build naturally on the previous one. The user asks follow-up questions. "
        "Keep assistant responses concise but informative (2-3 sentences each)."
    )

    prompt = f"""Create a {n_turns}-turn conversation about {category} (topic: {topic}).

The user starts with a question, the assistant answers, then the user asks a follow-up, etc.
Each exchange should be natural and build on the previous context.
Keep assistant answers concise (2-3 sentences each)."""

    for attempt in range(retry_count):
        try:
            result = call_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_schema=MULTI_TURN_SCHEMA,
                temperature=0.9,
                max_tokens=600
            )
            resp = result.get("response", {})
            if isinstance(resp, dict) and resp.get("turns"):
                turns = resp["turns"]
                if len(turns) >= 2:
                    # Convert to messages format
                    messages = []
                    for turn in turns:
                        if turn.get("user") and turn.get("assistant"):
                            messages.append({"role": "user", "content": turn["user"].strip()})
                            messages.append({"role": "assistant", "content": turn["assistant"].strip()})
                    if len(messages) >= 4:  # At least 2 turns
                        return {"messages": messages}
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(1)
            else:
                print(f"  [multi-turn] Failed ({category}): {e}")
    return None


def generate_cot(category: str, retry_count: int = 3) -> Optional[dict]:
    """Generate a chain-of-thought reasoning example."""
    topic = random.choice(TOPICS.get(category, ["general"]))

    cot_types = {
        "math": "a math problem that requires multi-step calculation",
        "science": "a science question that requires logical reasoning",
        "logic": "a logic puzzle or reasoning challenge",
        "programming": "a coding problem that requires step-by-step thinking",
        "general_knowledge": "a question that requires connecting multiple facts",
        "explanation": "a 'why' question that requires building an argument step by step",
        "history": "a historical cause-and-effect question",
        "geography": "a geography question requiring comparison or deduction",
        "language": "a language question requiring grammatical analysis",
        "creative": "a creative challenge requiring structured brainstorming",
    }

    problem_type = cot_types.get(category, "a question requiring step-by-step reasoning")

    system_prompt = (
        "You create questions that require step-by-step reasoning. "
        "The 'thinking' field should show clear reasoning steps (Step 1, Step 2, etc). "
        "The 'answer' field should be the final concise answer. "
        "Make the reasoning explicit and educational."
    )

    prompt = f"""Create {problem_type} about {topic}.

Requirements:
- The question should require 2-4 steps of reasoning
- Show clear step-by-step thinking (numbered steps)
- End with a concise final answer
- Make it educational and accessible"""

    for attempt in range(retry_count):
        try:
            result = call_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_schema=COT_SCHEMA,
                temperature=0.85,
                max_tokens=500
            )
            resp = result.get("response", {})
            if isinstance(resp, dict) and resp.get("instruction") and resp.get("thinking") and resp.get("answer"):
                instr = resp["instruction"].strip()
                thinking = resp["thinking"].strip()
                answer = resp["answer"].strip()
                if len(instr) > 10 and len(thinking) > 30 and len(answer) > 5:
                    # Format as instruction/output with thinking tags
                    output = f"<think>\n{thinking}\n</think>\n\n{answer}"
                    return {"instruction": instr, "input": "", "output": output}
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(1)
            else:
                print(f"  [cot] Failed ({category}): {e}")
    return None


def generate_spanish_qa(category: str, retry_count: int = 3) -> Optional[dict]:
    """Generate a single-turn Q&A pair in Spanish."""
    topic = random.choice(TOPICS.get(category, ["general"]))

    system_prompt = (
        "Eres un experto en crear pares de preguntas y respuestas educativas de alta calidad EN ESPAÑOL. "
        "Genera una pregunta realista y una respuesta clara y útil. "
        "Mantén las respuestas concisas (2-4 oraciones). Varía el estilo y la dificultad."
    )

    prompt = f"""Genera un par de pregunta y respuesta sobre {category} (tema: {topic}) EN ESPAÑOL.

La pregunta debe sonar natural, como si una persona real preguntara.
La respuesta debe ser precisa, educativa y concisa (2-4 oraciones).
TODO debe estar en español."""

    for attempt in range(retry_count):
        try:
            result = call_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_schema=SPANISH_QA_SCHEMA,
                temperature=0.9,
                max_tokens=300
            )
            resp = result.get("response", {})
            if isinstance(resp, dict) and resp.get("instruction") and resp.get("response"):
                instr = resp["instruction"].strip()
                ans = resp["response"].strip()
                if len(instr) > 10 and len(ans) > 20:
                    return {"instruction": instr, "input": "", "output": ans}
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(1)
            else:
                print(f"  [spanish-qa] Failed ({category}): {e}")
    return None


def generate_spanish_cot(category: str, retry_count: int = 3) -> Optional[dict]:
    """Generate a chain-of-thought reasoning example in Spanish."""
    topic = random.choice(TOPICS.get(category, ["general"]))

    system_prompt = (
        "Creas preguntas que requieren razonamiento paso a paso EN ESPAÑOL. "
        "El campo 'thinking' debe mostrar pasos claros de razonamiento (Paso 1, Paso 2, etc). "
        "El campo 'answer' debe ser la respuesta final concisa. "
        "Todo en español."
    )

    prompt = f"""Crea una pregunta sobre {category} (tema: {topic}) que requiera razonamiento paso a paso.

Requisitos:
- La pregunta debe requerir 2-4 pasos de razonamiento
- Muestra el razonamiento paso a paso (pasos numerados)
- Termina con una respuesta final concisa
- Todo debe estar EN ESPAÑOL"""

    for attempt in range(retry_count):
        try:
            result = call_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_schema=SPANISH_COT_SCHEMA,
                temperature=0.85,
                max_tokens=500
            )
            resp = result.get("response", {})
            if isinstance(resp, dict) and resp.get("instruction") and resp.get("thinking") and resp.get("answer"):
                instr = resp["instruction"].strip()
                thinking = resp["thinking"].strip()
                answer = resp["answer"].strip()
                if len(instr) > 10 and len(thinking) > 30 and len(answer) > 5:
                    output = f"<think>\n{thinking}\n</think>\n\n{answer}"
                    return {"instruction": instr, "input": "", "output": output}
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(1)
            else:
                print(f"  [spanish-cot] Failed ({category}): {e}")
    return None


# ── Distribution weights ────────────────────────────────────────────

# 30% single-turn EN, 20% multi-turn, 20% CoT EN, 15% Spanish QA, 15% Spanish CoT
GENERATORS = [
    ("single_turn", generate_single_turn, 30),
    ("multi_turn", generate_multi_turn, 20),
    ("cot", generate_cot, 20),
    ("spanish_qa", generate_spanish_qa, 15),
    ("spanish_cot", generate_spanish_cot, 15),
]


def pick_generator():
    """Pick a generator based on weights."""
    total = sum(w for _, _, w in GENERATORS)
    r = random.randint(1, total)
    cumulative = 0
    for name, gen_fn, weight in GENERATORS:
        cumulative += weight
        if r <= cumulative:
            return name, gen_fn
    return GENERATORS[0][0], GENERATORS[0][1]


# ── Main generation loop ───────────────────────────────────────────

def generate_batch(num_examples: int, output_path: str, num_workers: int = 1):
    """Generate a batch of mixed examples and save to JSONL (streaming).

    Uses ThreadPoolExecutor for concurrent API calls when num_workers > 1.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name, _, _ in GENERATORS}
    total_ok = 0
    total_failed = 0
    write_lock = threading.Lock()

    print(f"Generating {num_examples} mixed examples ({num_workers} workers)...")
    print(f"Distribution: {', '.join(f'{name}={w}%' for name, _, w in GENERATORS)}")
    print(f"Output: {output_path}")
    print()

    def do_one(_idx):
        """Generate a single example (thread-safe)."""
        gen_name, gen_fn = pick_generator()
        category = random.choice(CATEGORIES)
        result = gen_fn(category)
        return gen_name, result

    # Stream results to file (don't lose data on crash)
    with open(output_file, 'a', encoding='utf-8') as f:
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(do_one, i): i for i in range(num_examples)}
            for future in as_completed(futures):
                gen_name, result = future.result()
                completed += 1

                with write_lock:
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                        counts[gen_name] += 1
                        total_ok += 1
                    else:
                        total_failed += 1

                    if completed % 50 == 0:
                        parts = ", ".join(f"{k}={v}" for k, v in counts.items())
                        print(f"  [{completed}/{num_examples}] ok={total_ok} fail={total_failed} | {parts}")

    print(f"\nDone! Generated {total_ok} examples ({total_failed} failed)")
    print(f"Breakdown: {json.dumps(counts, indent=2)}")
    print(f"Saved to: {output_path}")

    return total_ok


def test_api():
    """Test the API connection."""
    print("Testing API connection...")
    try:
        result = call_completion(
            prompt="Say 'API working' in exactly 2 words",
            temperature=0.1,
            max_tokens=10
        )
        print(f"API Response: {result.get('response', 'No response')}")
        return True
    except Exception as e:
        print(f"API Error: {e}")
        print(f"Make sure the LLM API is running at {API_BASE}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic instruction data (multi-type)")
    parser.add_argument("--output", type=str, default="data/chat/synthetic_10k.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--num", type=int, default=10000,
                        help="Number of examples to generate")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent API workers (default: 4)")
    parser.add_argument("--test", action="store_true",
                        help="Test API connection only")
    args = parser.parse_args()

    if args.test:
        test_api()
        return

    if not test_api():
        print("\nPlease start the LLM API and try again.")
        return

    print()
    generate_batch(args.num, args.output, num_workers=args.workers)


if __name__ == "__main__":
    main()
