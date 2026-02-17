"""Generate English-only synthetic pretrain data with high concurrency.

Wraps the existing generator but filters to English-only types
and allows higher worker counts.

Usage:
    python scripts/generate_pretrain_english.py --num 50000 --workers 32
"""
import sys
from pathlib import Path

# Patch GENERATORS before importing the main function
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "generate_pretrain_data",
    Path(__file__).parent / "generate_pretrain_data.py"
)
gen_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_mod)

# Keep only English generators, rebalance weights
ENGLISH_GENERATORS = [
    ("tiny_knowledge", gen_mod.GENERATORS[0][1], 25),
    ("tiny_reasoning", gen_mod.GENERATORS[1][1], 25),
    ("tiny_facts",     gen_mod.GENERATORS[2][1], 15),
    ("tiny_dialogue",  gen_mod.GENERATORS[3][1], 15),
    ("tiny_code",      gen_mod.GENERATORS[4][1], 10),
]

# Patch the module
gen_mod.GENERATORS = ENGLISH_GENERATORS

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=str, default="data/synthetic_pretrain/train_english.txt")
    args = parser.parse_args()

    print(f"English-only generation: {args.num} docs, {args.workers} workers")
    print(f"Types: {[n for n,_,_ in ENGLISH_GENERATORS]}")
    gen_mod.generate_pretrain_data(args.num, args.output, args.workers)
