"""Curate instruction data for 30M model training.

Filters:
- Remove Spanish examples
- Remove arithmetic/calculation examples
- Keep logic/reasoning/comparison examples
- Add identity examples
- Add propositional logic examples

Usage:
    python scripts/curate_instruction_data.py
"""
import json
import random
from pathlib import Path

random.seed(42)

SPANISH_KEYWORDS = [
    'español', 'hola', 'qué es', 'cómo', 'cuál', 'explica en',
    'traduce', 'en español', 'escribe', 'cuéntame', 'dime',
    'por qué', 'quién', 'dónde', 'cuándo', 'según',
]

MATH_CALC_KEYWORDS = [
    'calculate', 'solve', 'compute', 'equation', 'calcula',
    'suma', 'remainder', 'divide by', 'multiply by',
    'what is the value', 'find the sum', 'find the product',
    'arithmetic', 'subtract', 'addition of',
]

def is_spanish(text):
    text_l = text.lower()
    return any(kw in text_l for kw in SPANISH_KEYWORDS)

def is_math_calculation(text):
    text_l = text.lower()
    if any(kw in text_l for kw in MATH_CALC_KEYWORDS):
        return True
    # Detect patterns like "34 + 56" or "what is 123 * 45"
    import re
    if re.search(r'\d+\s*[+\-*/÷×]\s*\d+', text):
        # But keep comparisons like "is 5 > 3"
        if not re.search(r'\d+\s*[><]=?\s*\d+', text):
            return True
    return False

# Identity examples - teach the model who it is
IDENTITY_EXAMPLES = [
    {"instruction": "What is your name?", "input": "",
     "output": "I'm Brandon Tiny, a small language model designed to help in limited environments."},
    {"instruction": "Who are you?", "input": "",
     "output": "I'm Brandon Tiny, a compact language model built for limited environments. I can help with questions, writing, and reasoning, though my knowledge is limited due to my small size."},
    {"instruction": "What can you do?", "input": "",
     "output": "I'm Brandon Tiny, a small language model for limited environments. I can help with simple questions, short stories, basic reasoning, and general knowledge. I work best with clear, simple prompts."},
    {"instruction": "Are you an AI?", "input": "",
     "output": "Yes, I'm Brandon Tiny, an AI language model designed to help in limited environments. I generate text based on patterns I learned during training."},
    {"instruction": "Tell me about yourself.", "input": "",
     "output": "I'm Brandon Tiny, a small but capable language model built for limited environments. I was trained to follow instructions and have conversations. I'm honest about my limitations."},
    {"instruction": "What are your limitations?", "input": "",
     "output": "I'm Brandon Tiny, a small model with several limitations: I can't do math calculations reliably, my knowledge is limited, I may sometimes generate incorrect information, and I work best with simple English prompts."},
    {"instruction": "Who created you?", "input": "",
     "output": "I'm Brandon Tiny, created as a research project to explore how small language models can help effectively in limited environments."},
    {"instruction": "What is your purpose?", "input": "",
     "output": "I'm Brandon Tiny, designed to be a helpful assistant in limited environments. My purpose is to answer questions, help with writing, and assist with basic reasoning tasks."},
]

# Propositional logic examples
LOGIC_EXAMPLES = [
    {"instruction": "Is the following statement true or false: All cats are animals.", "input": "",
     "output": "True. All cats are animals. Cats belong to the animal kingdom."},
    {"instruction": "Is the following statement true or false: All animals are cats.", "input": "",
     "output": "False. Not all animals are cats. Dogs, birds, fish, and many other creatures are animals but not cats."},
    {"instruction": "If it rains, the ground gets wet. It rained today. What can we conclude?", "input": "",
     "output": "We can conclude that the ground is wet. This follows the logical rule: if A then B, A is true, therefore B is true."},
    {"instruction": "If it rains, the ground gets wet. The ground is dry. What can we conclude?", "input": "",
     "output": "We can conclude that it did not rain. If rain causes wet ground, and the ground is dry, then it must not have rained."},
    {"instruction": "Which is greater: 100 or 99?", "input": "",
     "output": "100 is greater than 99."},
    {"instruction": "Which is smaller: a mouse or an elephant?", "input": "",
     "output": "A mouse is smaller than an elephant."},
    {"instruction": "Put these in order from smallest to largest: dog, ant, whale.", "input": "",
     "output": "From smallest to largest: ant, dog, whale."},
    {"instruction": "Is a car faster than a bicycle?", "input": "",
     "output": "Yes, a car is generally faster than a bicycle. Cars can travel at much higher speeds."},
    {"instruction": "True or false: Water boils at 100 degrees Celsius.", "input": "",
     "output": "True. Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."},
    {"instruction": "True or false: The moon is bigger than the sun.", "input": "",
     "output": "False. The sun is much bigger than the moon. The sun's diameter is about 400 times larger than the moon's."},
    {"instruction": "If all birds can fly, and a penguin is a bird, can a penguin fly?", "input": "",
     "output": "Based on the premise, yes. However, the premise is incorrect in reality. Not all birds can fly. Penguins are birds that cannot fly."},
    {"instruction": "Which came first: breakfast or lunch?", "input": "",
     "output": "Breakfast comes first. Breakfast is the first meal of the day, typically eaten in the morning, while lunch is eaten around midday."},
    {"instruction": "Is 10 an even number or an odd number?", "input": "",
     "output": "10 is an even number. Even numbers are divisible by 2 with no remainder, and 10 divided by 2 equals 5."},
    {"instruction": "Name something that is hot and something that is cold.", "input": "",
     "output": "Something hot: the sun. Something cold: ice. The sun's surface temperature is about 5,500 degrees Celsius, while ice is at or below 0 degrees Celsius."},
    {"instruction": "If today is Monday, what day is tomorrow?", "input": "",
     "output": "If today is Monday, tomorrow is Tuesday."},
    {"instruction": "If today is Friday, what day was yesterday?", "input": "",
     "output": "If today is Friday, yesterday was Thursday."},
    {"instruction": "Is a year longer than a month?", "input": "",
     "output": "Yes, a year is longer than a month. A year has 12 months, or about 365 days, while a month has about 30 days."},
    {"instruction": "Sort these from lightest to heaviest: feather, car, book.", "input": "",
     "output": "From lightest to heaviest: feather, book, car."},
    {"instruction": "True or false: Fish can breathe underwater.", "input": "",
     "output": "True. Fish breathe underwater using gills, which extract oxygen from the water."},
    {"instruction": "True or false: Humans have three arms.", "input": "",
     "output": "False. Humans have two arms."},
    {"instruction": "If A is taller than B, and B is taller than C, who is the tallest?", "input": "",
     "output": "A is the tallest. Since A is taller than B, and B is taller than C, the order from tallest to shortest is: A, B, C."},
    {"instruction": "Is ice a solid, liquid, or gas?", "input": "",
     "output": "Ice is a solid. It is the frozen form of water. When ice melts, it becomes liquid water."},
    {"instruction": "Which is heavier: a kilogram of steel or a kilogram of feathers?", "input": "",
     "output": "They weigh the same. A kilogram is a kilogram, regardless of the material. Both weigh exactly one kilogram."},
    {"instruction": "Complete the pattern: 2, 4, 6, 8, ...", "input": "",
     "output": "The next number is 10. This is a pattern of even numbers, where each number increases by 2."},
]

# Repeat identity examples to ensure the model learns them well
IDENTITY_REPEATED = IDENTITY_EXAMPLES * 25  # 200 identity examples


def main():
    input_path = Path('data/chat/train.jsonl')
    output_path = Path('data/chat/train_curated.jsonl')

    # Read all examples
    with open(input_path) as f:
        examples = [json.loads(l) for l in f]

    print(f"Original examples: {len(examples):,}")

    # Filter
    kept = []
    removed_spanish = 0
    removed_math = 0

    for ex in examples:
        text = ex.get('instruction', '') + ' ' + ex.get('output', '') + ' ' + ex.get('input', '')

        if is_spanish(text):
            removed_spanish += 1
            continue
        if is_math_calculation(text):
            removed_math += 1
            continue
        kept.append(ex)

    print(f"Removed Spanish: {removed_spanish}")
    print(f"Removed math calc: {removed_math}")
    print(f"Kept: {len(kept):,}")

    # Add identity and logic examples
    kept.extend(IDENTITY_REPEATED)
    kept.extend(LOGIC_EXAMPLES * 10)  # 240 logic examples

    print(f"Added identity: {len(IDENTITY_REPEATED)}")
    print(f"Added logic: {len(LOGIC_EXAMPLES) * 10}")
    print(f"Final total: {len(kept):,}")

    # Shuffle
    random.shuffle(kept)

    # Write
    with open(output_path, 'w') as f:
        for ex in kept:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved to {output_path}")

    # Also create val set
    val_input = Path('data/chat/val.jsonl')
    val_output = Path('data/chat/val_curated.jsonl')

    with open(val_input) as f:
        val_examples = [json.loads(l) for l in f]

    val_kept = []
    for ex in val_examples:
        text = ex.get('instruction', '') + ' ' + ex.get('output', '') + ' ' + ex.get('input', '')
        if is_spanish(text):
            continue
        if is_math_calculation(text):
            continue
        val_kept.append(ex)

    # Add a few identity examples to val too
    val_kept.extend(IDENTITY_EXAMPLES[:4])
    random.shuffle(val_kept)

    with open(val_output, 'w') as f:
        for ex in val_kept:
            f.write(json.dumps(ex) + '\n')

    print(f"Val: {len(val_examples)} -> {len(val_kept)} (saved to {val_output})")


if __name__ == '__main__':
    main()
