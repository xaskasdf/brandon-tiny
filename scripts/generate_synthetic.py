#!/usr/bin/env python
"""
Generate synthetic instruction data for TinyLlama fine-tuning.

Usage:
    python scripts/generate_synthetic.py --num_examples 5000 --output data/chat/train.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Expanded templates for synthetic data
TEMPLATES = {
    # Basic math
    "addition": {
        "instruction": "What is {a} + {b}?",
        "output": "The answer is {c}. {a} + {b} = {c}.",
        "gen": lambda: {"a": random.randint(1, 100), "b": random.randint(1, 100)},
        "compute": lambda d: {"c": d["a"] + d["b"]}
    },
    "subtraction": {
        "instruction": "Calculate {a} - {b}.",
        "output": "{a} minus {b} equals {c}.",
        "gen": lambda: {"a": random.randint(50, 200), "b": random.randint(1, 50)},
        "compute": lambda d: {"c": d["a"] - d["b"]}
    },
    "multiplication": {
        "instruction": "What is {a} times {b}?",
        "output": "{a} × {b} = {c}.",
        "gen": lambda: {"a": random.randint(2, 15), "b": random.randint(2, 15)},
        "compute": lambda d: {"c": d["a"] * d["b"]}
    },
    "division": {
        "instruction": "Divide {a} by {b}.",
        "output": "{a} ÷ {b} = {c}.",
        "gen": lambda: (lambda b, m: {"a": b * m, "b": b})(random.randint(2, 10), random.randint(1, 10)),
        "compute": lambda d: {"c": d["a"] // d["b"]}
    },

    # Number properties
    "even_odd": {
        "instruction": "Is {n} even or odd?",
        "output": "{n} is {parity}.",
        "gen": lambda: {"n": random.randint(1, 1000)},
        "compute": lambda d: {"parity": "even" if d["n"] % 2 == 0 else "odd"}
    },
    "divisible": {
        "instruction": "Is {a} divisible by {b}?",
        "output": "{answer}",
        "gen": lambda: {"a": random.randint(10, 100), "b": random.choice([2, 3, 5, 10])},
        "compute": lambda d: {"answer": f"Yes, {d['a']} is divisible by {d['b']}." if d['a'] % d['b'] == 0 else f"No, {d['a']} is not divisible by {d['b']}."}
    },

    # Comparisons
    "compare": {
        "instruction": "Which is larger: {a} or {b}?",
        "output": "{answer}",
        "gen": lambda: {"a": random.randint(1, 1000), "b": random.randint(1, 1000)},
        "compute": lambda d: {"answer": f"{d['a']} is larger." if d['a'] > d['b'] else f"{d['b']} is larger." if d['b'] > d['a'] else "They are equal."}
    },

    # Unit conversions
    "meters_to_cm": {
        "instruction": "Convert {n} meters to centimeters.",
        "output": "{n} meters = {cm} centimeters.",
        "gen": lambda: {"n": random.randint(1, 100)},
        "compute": lambda d: {"cm": d["n"] * 100}
    },
    "km_to_m": {
        "instruction": "Convert {n} kilometers to meters.",
        "output": "{n} kilometers = {m} meters.",
        "gen": lambda: {"n": random.randint(1, 50)},
        "compute": lambda d: {"m": d["n"] * 1000}
    },
    "hours_to_minutes": {
        "instruction": "How many minutes are in {n} hours?",
        "output": "{n} hours = {m} minutes.",
        "gen": lambda: {"n": random.randint(1, 24)},
        "compute": lambda d: {"m": d["n"] * 60}
    },

    # Simple facts
    "capital": {
        "instruction": "What is the capital of {country}?",
        "output": "The capital of {country} is {capital}.",
        "gen": lambda: random.choice([
            {"country": "France", "capital": "Paris"},
            {"country": "Japan", "capital": "Tokyo"},
            {"country": "Germany", "capital": "Berlin"},
            {"country": "Italy", "capital": "Rome"},
            {"country": "Spain", "capital": "Madrid"},
            {"country": "United Kingdom", "capital": "London"},
            {"country": "China", "capital": "Beijing"},
            {"country": "Brazil", "capital": "Brasília"},
        ]),
        "compute": lambda d: {}
    },
    "color": {
        "instruction": "What color is {thing}?",
        "output": "{thing} is typically {color}.",
        "gen": lambda: random.choice([
            {"thing": "the sky", "color": "blue"},
            {"thing": "grass", "color": "green"},
            {"thing": "the sun", "color": "yellow"},
            {"thing": "snow", "color": "white"},
            {"thing": "coal", "color": "black"},
            {"thing": "a tomato", "color": "red"},
        ]),
        "compute": lambda d: {}
    },
    "days_in_month": {
        "instruction": "How many days are in {month}?",
        "output": "{month} has {days} days.",
        "gen": lambda: random.choice([
            {"month": "January", "days": 31},
            {"month": "February", "days": "28 or 29"},
            {"month": "March", "days": 31},
            {"month": "April", "days": 30},
            {"month": "May", "days": 31},
            {"month": "June", "days": 30},
            {"month": "July", "days": 31},
            {"month": "August", "days": 31},
            {"month": "September", "days": 30},
            {"month": "October", "days": 31},
            {"month": "November", "days": 30},
            {"month": "December", "days": 31},
        ]),
        "compute": lambda d: {}
    },

    # Counting
    "count_letters": {
        "instruction": "How many letters are in the word '{word}'?",
        "output": "The word '{word}' has {count} letters.",
        "gen": lambda: {"word": random.choice([
            "cat", "dog", "house", "computer", "python", "language",
            "programming", "artificial", "intelligence", "machine"
        ])},
        "compute": lambda d: {"count": len(d["word"])}
    },

    # Greetings
    "greeting": {
        "instruction": "{greeting}",
        "output": "{response}",
        "gen": lambda: random.choice([
            {"greeting": "Hello!", "response": "Hello! How can I help you today?"},
            {"greeting": "Hi there", "response": "Hi! What can I do for you?"},
            {"greeting": "Good morning", "response": "Good morning! How may I assist you?"},
            {"greeting": "Hey", "response": "Hey! What's on your mind?"},
        ]),
        "compute": lambda d: {}
    },

    # Simple reasoning
    "word_problem_add": {
        "instruction": "{name} has {a} apples. {name2} gives {name} {b} more apples. How many apples does {name} have now?",
        "output": "{name} now has {c} apples. ({a} + {b} = {c})",
        "gen": lambda: {
            "name": random.choice(["Alice", "Bob", "Charlie", "Diana"]),
            "name2": random.choice(["Emma", "Frank", "Grace", "Henry"]),
            "a": random.randint(1, 20),
            "b": random.randint(1, 20)
        },
        "compute": lambda d: {"c": d["a"] + d["b"]}
    },
}


def generate_example(template_name: str) -> dict:
    """Generate a single example from a template."""
    template = TEMPLATES[template_name]
    data = template["gen"]()
    data.update(template["compute"](data))

    return {
        "instruction": template["instruction"].format(**data),
        "output": template["output"].format(**data)
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic instruction data")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5000,
        help="Number of examples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chat/train.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Generate examples
    print(f"Generating {args.num_examples} examples...")
    examples = []
    template_names = list(TEMPLATES.keys())

    for _ in range(args.num_examples):
        template_name = random.choice(template_names)
        example = generate_example(template_name)
        examples.append(example)

    # Shuffle
    random.shuffle(examples)

    # Split into train/val
    val_size = int(len(examples) * args.val_split)
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"Saved {len(train_examples)} training examples to {output_path}")

    # Save validation set
    val_path = output_path.parent / "val.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"Saved {len(val_examples)} validation examples to {val_path}")

    # Print statistics
    print("\nTemplate distribution:")
    from collections import Counter
    template_counts = Counter()
    for ex in examples:
        for name, tmpl in TEMPLATES.items():
            if tmpl["instruction"].split()[0] in ex["instruction"]:
                template_counts[name] += 1
                break

    for name, count in template_counts.most_common(10):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
