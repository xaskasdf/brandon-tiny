#!/usr/bin/env python
"""
Generate synthetic pre-training data using a local LLM API.

Produces high-quality "textbook-style" text (Phi approach) in several formats:
  - TinyKnowledge: Clear explanations of concepts (encyclopedia-style)
  - TinyReasoning: Problems with step-by-step solutions in prose
  - TinyFacts: Collections of related facts about a topic
  - TinyDialogues: Educational conversations in narrative form
  - TinyCode: Simple code with natural language explanations
  - TinySpanish: All types in Spanish

Output is plain text (one document per line) for pre-training, NOT instruction format.

Usage:
    python scripts/generate_pretrain_data.py --output data/synthetic_pretrain/train.txt --num 5000
    python scripts/generate_pretrain_data.py --output data/synthetic_pretrain/train.txt --num 5000 --workers 4
    python scripts/generate_pretrain_data.py --type tiny_reasoning --num 1000
    python scripts/generate_pretrain_data.py --test
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

API_BASE = "http://localhost:5282"


def call_completion(prompt: str, system_prompt: str = None, temperature: float = 0.85,
                    max_tokens: int = 600) -> Optional[str]:
    """Call the LLM API and return raw text response."""
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt

    try:
        resp = requests.post(f"{API_BASE}/completion", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        if isinstance(text, str) and len(text.strip()) > 50:
            return text.strip()
    except Exception:
        pass
    return None


# ── Topics ────────────────────────────────────────────────────────

KNOWLEDGE_TOPICS = [
    # Science
    "photosynthesis", "gravity", "the water cycle", "magnets", "electricity",
    "the solar system", "DNA and genes", "volcanoes", "earthquakes", "sound waves",
    "light and colors", "atoms and molecules", "the human heart", "ecosystems",
    "weather and climate", "the moon", "stars and constellations", "bacteria",
    "the ocean", "fossils", "dinosaurs", "the atmosphere", "nutrition",
    # Math
    "prime numbers", "fractions", "geometry basics", "probability", "percentages",
    "negative numbers", "the Pythagorean theorem", "area and perimeter", "graphs",
    # History
    "ancient Egypt", "the Roman Empire", "the printing press", "the Industrial Revolution",
    "world exploration", "ancient Greece", "medieval castles", "the Renaissance",
    # Geography
    "mountains of the world", "rivers and lakes", "deserts", "tropical rainforests",
    "the Arctic and Antarctic", "islands", "oceans and seas",
    # Technology
    "how computers work", "the internet", "robots", "artificial intelligence basics",
    "how phones work", "databases", "programming languages", "algorithms",
    # General
    "how bridges are built", "musical instruments", "how planes fly",
    "the history of writing", "how maps are made", "photography basics",
]

REASONING_PROBLEMS = [
    "a math word problem about shopping and calculating change",
    "a logic puzzle about who sits where at a table",
    "a probability question about drawing colored balls from a bag",
    "a geometry problem about finding areas of shapes",
    "a sequence/pattern recognition problem",
    "a problem about mixing ingredients in different ratios",
    "a time and distance calculation problem",
    "a problem about sharing items fairly among friends",
    "a cause-and-effect reasoning chain about nature",
    "a comparison problem requiring multi-step reasoning",
    "a scheduling problem with constraints",
    "a problem about growth and percentages over time",
    "a spatial reasoning problem about arranging objects",
    "a deductive reasoning puzzle with clues",
    "an estimation problem requiring approximation skills",
]

CODE_TOPICS = [
    "a function that checks if a number is prime",
    "a function that reverses a string",
    "a loop that finds the largest number in a list",
    "a simple calculator that adds two numbers from user input",
    "a function that counts vowels in a word",
    "a program that prints a multiplication table",
    "a function that checks if a word is a palindrome",
    "a function that converts temperature between Celsius and Fahrenheit",
    "a simple sorting algorithm (bubble sort)",
    "a function that finds common elements in two lists",
    "a program that generates Fibonacci numbers",
    "a function that calculates the factorial of a number",
    "a simple text-based guessing game",
    "a function that removes duplicates from a list",
    "a program that counts word frequency in a sentence",
]

DIALOGUE_SCENARIOS = [
    "a student asking a teacher about how plants grow",
    "a child asking a parent why the sky is blue",
    "two friends discussing how computers work",
    "a curious person learning about space from an astronomer",
    "a student and teacher exploring basic chemistry",
    "two people discussing the history of their city",
    "a child learning about animals in the jungle",
    "a new programmer asking an expert about variables and loops",
    "two friends figuring out a math puzzle together",
    "a student asking about how electricity works",
    "a child curious about why we need to sleep",
    "two people discussing how bridges stay up",
    "a learner asking about the difference between weather and climate",
    "a student and teacher discussing how music works",
    "two friends exploring why some things float and others sink",
]


# ── Generators ────────────────────────────────────────────────────

def gen_tiny_knowledge(lang="en") -> Optional[str]:
    """Generate encyclopedia-style explanation."""
    topic = random.choice(KNOWLEDGE_TOPICS)

    if lang == "es":
        system = ("Eres un escritor de enciclopedia educativa. Escribes explicaciones claras, "
                  "precisas y accesibles EN ESPAÑOL. Usa lenguaje simple pero correcto.")
        prompt = (f"Escribe una explicación clara y educativa sobre: {topic}\n\n"
                  "Requisitos:\n"
                  "- 3-5 párrafos\n"
                  "- Lenguaje simple y claro\n"
                  "- Incluye al menos un dato interesante\n"
                  "- Explicación autocontenida (no necesita contexto externo)\n"
                  "- Todo en español")
    else:
        system = ("You are an educational encyclopedia writer. You write clear, accurate, "
                  "and accessible explanations. Use simple but precise language.")
        prompt = (f"Write a clear, educational explanation about: {topic}\n\n"
                  "Requirements:\n"
                  "- 3-5 paragraphs\n"
                  "- Simple, clear language (suitable for a curious teenager)\n"
                  "- Include at least one interesting fact or surprising detail\n"
                  "- Self-contained (no external context needed)\n"
                  "- End with a thought-provoking question or connection to daily life")

    return call_completion(prompt, system, temperature=0.85, max_tokens=500)


def gen_tiny_reasoning(lang="en") -> Optional[str]:
    """Generate a reasoning problem with step-by-step solution in prose."""
    problem_type = random.choice(REASONING_PROBLEMS)

    if lang == "es":
        system = ("Creas problemas de razonamiento con soluciones paso a paso EN ESPAÑOL. "
                  "Escribes como un libro de texto educativo.")
        prompt = (f"Crea {problem_type}.\n\n"
                  "Formato (como texto continuo, NO como chat):\n"
                  "1. Presenta el problema claramente\n"
                  "2. Muestra la solución paso a paso\n"
                  "3. Explica el razonamiento en cada paso\n"
                  "4. Da la respuesta final\n"
                  "Todo en español, como un libro de texto.")
    else:
        system = ("You create reasoning problems with step-by-step solutions. "
                  "Write like an educational textbook - clear prose, not chat format.")
        prompt = (f"Create {problem_type}.\n\n"
                  "Format (as continuous prose, NOT as chat):\n"
                  "1. Present the problem clearly\n"
                  "2. Show the step-by-step solution\n"
                  "3. Explain the reasoning at each step\n"
                  "4. State the final answer\n"
                  "Write it like a textbook example with explanation.")

    return call_completion(prompt, system, temperature=0.8, max_tokens=600)


def gen_tiny_facts(lang="en") -> Optional[str]:
    """Generate a collection of related facts about a topic."""
    topic = random.choice(KNOWLEDGE_TOPICS)

    if lang == "es":
        system = "Eres un escritor de datos curiosos educativos EN ESPAÑOL."
        prompt = (f"Escribe una colección de 8-12 datos interesantes y educativos sobre: {topic}\n\n"
                  "Formato: texto continuo con párrafos cortos, cada dato conectado al siguiente. "
                  "No uses listas con viñetas. Escribe como un ensayo corto e informativo. "
                  "Todo en español.")
    else:
        system = "You are a writer of educational fun facts."
        prompt = (f"Write a collection of 8-12 interesting, educational facts about: {topic}\n\n"
                  "Format: continuous prose with short paragraphs, each fact flowing into the next. "
                  "Do NOT use bullet points. Write like a short, engaging informational essay. "
                  "Make connections between facts where possible.")

    return call_completion(prompt, system, temperature=0.9, max_tokens=500)


def gen_tiny_dialogue(lang="en") -> Optional[str]:
    """Generate educational dialogue in narrative form."""
    scenario = random.choice(DIALOGUE_SCENARIOS)

    if lang == "es":
        system = ("Escribes diálogos educativos en formato narrativo EN ESPAÑOL. "
                  "Los diálogos son naturales y enseñan conceptos de forma accesible.")
        prompt = (f"Escribe un diálogo educativo sobre: {scenario}\n\n"
                  "Formato narrativo (no chat):\n"
                  '- Usa comillas para el diálogo: "¿Por qué...?" preguntó María.\n'
                  "- 8-15 intercambios\n"
                  "- El experto explica con paciencia y ejemplos simples\n"
                  "- El aprendiz hace preguntas de seguimiento naturales\n"
                  "Todo en español.")
    else:
        system = ("You write educational dialogues in narrative form. "
                  "The dialogues feel natural and teach concepts accessibly.")
        prompt = (f"Write an educational dialogue about: {scenario}\n\n"
                  "Use narrative format (not chat):\n"
                  '- Use quotes: "Why does...?" asked Sarah.\n'
                  "- 8-15 exchanges\n"
                  "- The expert explains patiently with simple examples\n"
                  "- The learner asks natural follow-up questions\n"
                  "- Include moments of realization and connection-making")

    return call_completion(prompt, system, temperature=0.85, max_tokens=700)


def gen_tiny_code(lang="en") -> Optional[str]:
    """Generate simple code with natural language explanation."""
    topic = random.choice(CODE_TOPICS)

    if lang == "es":
        system = ("Eres un tutor de programación que explica código de forma clara EN ESPAÑOL. "
                  "Escribes como un libro de texto de introducción a la programación.")
        prompt = (f"Explica cómo escribir {topic} en Python.\n\n"
                  "Formato (como texto de libro):\n"
                  "1. Explica qué queremos lograr\n"
                  "2. Muestra el código\n"
                  "3. Explica cada parte del código línea por línea\n"
                  "4. Muestra un ejemplo de uso con su salida\n"
                  "Todo en español.")
    else:
        system = ("You are a programming tutor who explains code clearly. "
                  "Write like an introductory programming textbook.")
        prompt = (f"Explain how to write {topic} in Python.\n\n"
                  "Format (like a textbook section):\n"
                  "1. Explain what we want to achieve\n"
                  "2. Show the code\n"
                  "3. Walk through each part of the code\n"
                  "4. Show an example usage with expected output\n"
                  "Keep explanations simple and clear.")

    return call_completion(prompt, system, temperature=0.8, max_tokens=600)


# ── Distribution ──────────────────────────────────────────────────

# Weights: knowledge=25, reasoning=25, facts=15, dialogue=15, code=10, spanish=10
GENERATORS = [
    ("tiny_knowledge", lambda: gen_tiny_knowledge("en"), 20),
    ("tiny_reasoning", lambda: gen_tiny_reasoning("en"), 20),
    ("tiny_facts",     lambda: gen_tiny_facts("en"), 12),
    ("tiny_dialogue",  lambda: gen_tiny_dialogue("en"), 12),
    ("tiny_code",      lambda: gen_tiny_code("en"), 8),
    ("tiny_knowledge_es", lambda: gen_tiny_knowledge("es"), 8),
    ("tiny_reasoning_es", lambda: gen_tiny_reasoning("es"), 8),
    ("tiny_facts_es",     lambda: gen_tiny_facts("es"), 4),
    ("tiny_dialogue_es",  lambda: gen_tiny_dialogue("es"), 4),
    ("tiny_code_es",      lambda: gen_tiny_code("es"), 4),
]


def pick_generator(gen_type=None):
    """Pick a generator based on weights or specific type."""
    if gen_type:
        for name, fn, _ in GENERATORS:
            if name == gen_type:
                return name, fn
        raise ValueError(f"Unknown type: {gen_type}. Available: {[n for n,_,_ in GENERATORS]}")

    total = sum(w for _, _, w in GENERATORS)
    r = random.randint(1, total)
    cumulative = 0
    for name, fn, weight in GENERATORS:
        cumulative += weight
        if r <= cumulative:
            return name, fn
    return GENERATORS[0][0], GENERATORS[0][1]


# ── Main loop ─────────────────────────────────────────────────────

def generate_pretrain_data(num_docs: int, output_path: str, num_workers: int = 1,
                           gen_type: str = None):
    """Generate synthetic pre-training documents."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name, _, _ in GENERATORS}
    total_ok = 0
    total_failed = 0
    write_lock = threading.Lock()

    print(f"Generating {num_docs} pre-training documents ({num_workers} workers)...")
    if gen_type:
        print(f"Type filter: {gen_type}")
    else:
        dist = ", ".join(f"{n}={w}%" for n, _, w in GENERATORS)
        print(f"Distribution: {dist}")
    print(f"Output: {output_path}\n")

    def do_one(_idx):
        name, fn = pick_generator(gen_type)
        result = fn()
        return name, result

    with open(output_file, 'a', encoding='utf-8') as f:
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(do_one, i): i for i in range(num_docs)}
            for future in as_completed(futures):
                name, result = future.result()
                completed += 1

                with write_lock:
                    if result:
                        # Write as single line (newlines replaced with spaces within doc)
                        # Documents separated by double newline
                        clean = result.replace('\r\n', '\n')
                        f.write(clean + "\n\n")
                        f.flush()
                        counts[name] += 1
                        total_ok += 1
                    else:
                        total_failed += 1

                    if completed % 25 == 0:
                        parts = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
                        print(f"  [{completed}/{num_docs}] ok={total_ok} fail={total_failed} | {parts}")

    print(f"\nDone! Generated {total_ok} documents ({total_failed} failed)")
    print(f"Breakdown: {json.dumps({k:v for k,v in counts.items() if v > 0}, indent=2)}")

    # Show file size
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"File: {output_path} ({size_mb:.1f} MB)")

    return total_ok


def test_api():
    """Test API connection."""
    print("Testing API...")
    try:
        payload = {"prompt": "Explain what water is in 2 sentences.", "temperature": 0.1, "max_tokens": 50}
        resp = requests.post(f"{API_BASE}/completion", json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json().get("response", "")
        if text:
            print(f"API OK: {str(text)[:80]}")
            return True
        print("API returned empty response")
        return False
    except Exception as e:
        print(f"API Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pre-training data")
    parser.add_argument("--output", type=str, default="data/synthetic_pretrain/train.txt")
    parser.add_argument("--num", type=int, default=5000, help="Number of documents")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent API workers")
    parser.add_argument("--type", type=str, default=None,
                        help="Generate only this type (e.g. tiny_reasoning)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        test_api()
        return

    if not test_api():
        print("\nStart the LLM API and try again.")
        return

    print()
    generate_pretrain_data(args.num, args.output, args.workers, args.type)


if __name__ == "__main__":
    main()
