#!/usr/bin/env python
"""
Generate high-quality synthetic instruction data using a local LLM API.

Usage:
    python scripts/generate_synthetic_llm.py --output data/chat/train.jsonl --num 5000
    python scripts/generate_synthetic_llm.py --output data/chat/val.jsonl --num 500
"""

import argparse
import json
import random
import time
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

    response = requests.post(f"{API_BASE}/completion", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


# Categories for diverse instruction data
CATEGORIES = [
    "math",
    "science",
    "history",
    "geography",
    "language",
    "programming",
    "logic",
    "general_knowledge",
    "creative",
    "explanation",
]

# Templates for generating diverse prompts
PROMPT_TEMPLATES = {
    "math": [
        "Create a simple arithmetic problem with its solution",
        "Generate a word problem involving {topic} with step-by-step solution",
        "Create a basic algebra question suitable for beginners",
        "Generate a question about fractions or percentages with answer",
    ],
    "science": [
        "Create a question about {topic} with a clear, educational answer",
        "Generate a 'how does X work' question about everyday science",
        "Create a question about the human body with answer",
        "Generate a question about animals or nature with answer",
    ],
    "history": [
        "Create a question about an important historical event with answer",
        "Generate a 'who was' question about a famous historical figure",
        "Create a question about ancient civilizations with answer",
        "Generate a question about {topic} in history with answer",
    ],
    "geography": [
        "Create a question about countries, capitals, or landmarks",
        "Generate a question about {topic} geography with answer",
        "Create a question about world geography with answer",
        "Generate a question about continents or oceans with answer",
    ],
    "language": [
        "Create a vocabulary question with definition and example",
        "Generate a grammar question with explanation",
        "Create a question about synonyms or antonyms with answer",
        "Generate a simple translation or language learning question",
    ],
    "programming": [
        "Create a beginner programming question about {topic} with answer",
        "Generate a 'what is' question about a programming concept",
        "Create a simple coding question with solution",
        "Generate a question about debugging or best practices",
    ],
    "logic": [
        "Create a simple logic puzzle with solution",
        "Generate a pattern recognition question with answer",
        "Create a riddle with answer and explanation",
        "Generate a 'what comes next' sequence question",
    ],
    "general_knowledge": [
        "Create a trivia question with answer",
        "Generate a 'did you know' style question with answer",
        "Create a question about everyday life with helpful answer",
        "Generate a practical question about {topic} with answer",
    ],
    "creative": [
        "Create a short story prompt with a brief example continuation",
        "Generate a creative writing question with sample answer",
        "Create a 'describe' prompt with imaginative response",
        "Generate a brainstorming question with several ideas",
    ],
    "explanation": [
        "Create a 'explain like I'm 5' question about {topic}",
        "Generate a 'why' question with clear explanation",
        "Create a comparison question (X vs Y) with answer",
        "Generate a 'how to' question with step-by-step answer",
    ],
}

TOPICS = {
    "math": ["addition", "subtraction", "multiplication", "division", "percentages"],
    "science": ["physics", "chemistry", "biology", "astronomy", "weather"],
    "history": ["wars", "inventions", "empires", "revolutions", "discoveries"],
    "geography": ["mountains", "rivers", "deserts", "islands", "climate"],
    "programming": ["Python", "JavaScript", "algorithms", "data structures", "web development"],
    "general_knowledge": ["food", "sports", "music", "art", "technology"],
    "explanation": ["technology", "nature", "society", "health", "economics"],
}

# JSON schema for structured generation
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


def generate_qa_pair(category: str, retry_count: int = 3) -> Optional[dict]:
    """Generate a single Q&A pair using the LLM."""
    templates = PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["general_knowledge"])
    template = random.choice(templates)

    # Fill in topic if needed
    if "{topic}" in template:
        topics = TOPICS.get(category, ["general"])
        topic = random.choice(topics)
        template = template.replace("{topic}", topic)

    system_prompt = """You are an expert at creating high-quality educational Q&A pairs.
Generate a realistic question that a curious person might ask, and provide a clear,
helpful, and accurate answer. The answer should be informative but concise (2-4 sentences).
Keep the language simple and accessible."""

    prompt = f"""Generate a Q&A pair for this category: {category}

Task: {template}

Requirements:
- The question should sound natural, like a real person asking
- The answer should be accurate, helpful, and educational
- Keep the answer concise (2-4 sentences)
- Vary the style and complexity"""

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

            response = result.get("response", {})
            if isinstance(response, dict) and "instruction" in response and "response" in response:
                # Validate content
                instruction = response["instruction"].strip()
                answer = response["response"].strip()

                if len(instruction) > 10 and len(answer) > 20:
                    return {
                        "instruction": instruction,
                        "input": "",
                        "output": answer
                    }

        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(1)
            else:
                print(f"Failed to generate for {category}: {e}")

    return None


def generate_batch(num_examples: int, output_path: str, batch_size: int = 10):
    """Generate a batch of Q&A pairs and save to JSONL."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    failed = 0

    print(f"Generating {num_examples} examples...")
    print(f"Output: {output_path}")

    for i in range(num_examples):
        category = random.choice(CATEGORIES)

        qa_pair = generate_qa_pair(category)

        if qa_pair:
            examples.append(qa_pair)
            if (i + 1) % batch_size == 0:
                print(f"Progress: {i + 1}/{num_examples} ({len(examples)} successful)")
        else:
            failed += 1

        # Small delay to avoid overwhelming the API
        if (i + 1) % 5 == 0:
            time.sleep(0.1)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"\nDone! Generated {len(examples)} examples ({failed} failed)")
    print(f"Saved to: {output_path}")

    return examples


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
    parser = argparse.ArgumentParser(description="Generate synthetic instruction data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/chat/train_synthetic.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        help="Number of examples to generate"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test API connection only"
    )
    args = parser.parse_args()

    if args.test:
        test_api()
        return

    # Test API first
    if not test_api():
        print("\nPlease start the LLM API and try again.")
        return

    print()
    generate_batch(args.num, args.output)


if __name__ == "__main__":
    main()
