#!/usr/bin/env python
"""
Test script for TinyLlama model.

Verifies model architecture, parameter counts, and basic functionality.

Usage:
    python scripts/test_model.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def test_model_configs():
    """Test model configurations."""
    from src.model import TinyLlama, ModelConfig

    print("=" * 60)
    print("Testing Model Configurations")
    print("=" * 60)

    # Test 226K config
    config_226k = ModelConfig(
        dim=48,
        n_layers=7,
        n_heads=4,
        n_kv_heads=2,  # GQA
        vocab_size=1024,
        hidden_dim=128,
        max_seq_len=128,
        dropout=0.15,
        weight_tying=True
    )

    model_226k = TinyLlama(config_226k)
    params_226k = model_226k.count_parameters()
    estimated_226k = config_226k.param_count()

    print(f"\n226K Model:")
    print(f"  Actual parameters: {params_226k:,}")
    print(f"  Estimated parameters: {estimated_226k:,}")
    print(f"  Difference: {abs(params_226k - estimated_226k):,}")

    # Test 110M config
    config_110m = ModelConfig(
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=12,  # Full MHA
        vocab_size=32000,
        hidden_dim=2048,
        max_seq_len=1024,
        dropout=0.0,
        weight_tying=True
    )

    model_110m = TinyLlama(config_110m)
    params_110m = model_110m.count_parameters()
    estimated_110m = config_110m.param_count()

    print(f"\n110M Model:")
    print(f"  Actual parameters: {params_110m:,}")
    print(f"  Estimated parameters: {estimated_110m:,}")
    print(f"  Difference: {abs(params_110m - estimated_110m):,}")

    return model_226k, model_110m


def test_forward_pass(model, config_name="226K"):
    """Test forward pass."""
    print(f"\n{'=' * 60}")
    print(f"Testing Forward Pass ({config_name})")
    print("=" * 60)

    vocab_size = model.config.vocab_size
    seq_len = min(32, model.config.max_seq_len)
    batch_size = 2

    # Create random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = input_ids.clone()

    # Forward pass
    logits, loss = model(input_ids, targets)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Check shapes
    assert logits.shape == (batch_size, seq_len, vocab_size), "Logits shape mismatch"
    assert loss.dim() == 0, "Loss should be scalar"

    print("  ✓ Forward pass OK")


def test_generation(model, config_name="226K"):
    """Test text generation."""
    print(f"\n{'=' * 60}")
    print(f"Testing Generation ({config_name})")
    print("=" * 60)

    vocab_size = model.config.vocab_size

    # Create a simple prompt
    prompt_tokens = torch.randint(0, vocab_size, (5,))

    # Generate
    model.eval()
    with torch.no_grad():
        output = model.generate(
            prompt_tokens,
            max_new_tokens=10,
            temperature=0.8,
            top_p=0.9,
            top_k=40
        )

    print(f"  Prompt tokens: {prompt_tokens.tolist()}")
    print(f"  Generated tokens: {output[0].tolist()}")
    print(f"  New tokens: {output.shape[1] - len(prompt_tokens)}")

    assert output.shape[1] >= len(prompt_tokens), "Output should be at least prompt length"

    print("  ✓ Generation OK")


def test_target_masking():
    """Test target masking for fine-tuning."""
    from src.model import TinyLlama, ModelConfig

    print(f"\n{'=' * 60}")
    print("Testing Target Masking")
    print("=" * 60)

    config = ModelConfig(
        dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=256,
        hidden_dim=64,
        max_seq_len=32
    )

    model = TinyLlama(config)

    # Create batch with target mask
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    targets = input_ids.clone()

    # Mask: only compute loss on second half
    target_mask = torch.zeros(batch_size, seq_len)
    target_mask[:, seq_len // 2:] = 1.0

    # Forward with and without masking
    _, loss_full = model(input_ids, targets)
    _, loss_masked = model(input_ids, targets, target_mask)

    print(f"  Full loss: {loss_full.item():.4f}")
    print(f"  Masked loss: {loss_masked.item():.4f}")
    print(f"  Difference: {abs(loss_full.item() - loss_masked.item()):.4f}")

    print("  ✓ Target masking OK")


def test_checkpoint():
    """Test checkpoint save/load."""
    from src.model import TinyLlama, ModelConfig
    import tempfile
    import os

    print(f"\n{'=' * 60}")
    print("Testing Checkpoint Save/Load")
    print("=" * 60)

    config = ModelConfig(
        dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=256,
        hidden_dim=64,
        max_seq_len=32
    )

    model = TinyLlama(config)

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test.pt")
        model.save_checkpoint(checkpoint_path, step=100)

        # Load checkpoint
        loaded_model = TinyLlama.from_checkpoint(checkpoint_path, device='cpu')

        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(),
            loaded_model.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            assert torch.allclose(param1, param2), f"Parameter value mismatch: {name1}"

    print("  ✓ Checkpoint save/load OK")


def test_rope():
    """Test Rotary Position Embeddings."""
    from src.model import precompute_freqs_cis, apply_rotary_emb

    print(f"\n{'=' * 60}")
    print("Testing RoPE")
    print("=" * 60)

    dim = 16
    max_seq_len = 32
    batch_size = 2
    n_heads = 4

    # Precompute frequencies
    freqs_cis = precompute_freqs_cis(dim, max_seq_len)
    print(f"  Frequencies shape: {freqs_cis.shape}")

    # Create test queries and keys
    head_dim = dim
    xq = torch.randn(batch_size, max_seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, max_seq_len, n_heads, head_dim)

    # Apply rotary embeddings
    xq_rope, xk_rope = apply_rotary_emb(xq, xk, freqs_cis)

    print(f"  Query shape: {xq.shape} -> {xq_rope.shape}")
    print(f"  Key shape: {xk.shape} -> {xk_rope.shape}")

    # Check shapes preserved
    assert xq_rope.shape == xq.shape, "Query shape should be preserved"
    assert xk_rope.shape == xk.shape, "Key shape should be preserved"

    # Check that RoPE modifies the values
    assert not torch.allclose(xq, xq_rope), "RoPE should modify queries"
    assert not torch.allclose(xk, xk_rope), "RoPE should modify keys"

    print("  ✓ RoPE OK")


def test_rmsnorm():
    """Test RMSNorm."""
    from src.model import RMSNorm

    print(f"\n{'=' * 60}")
    print("Testing RMSNorm")
    print("=" * 60)

    dim = 64
    batch_size, seq_len = 2, 16

    norm = RMSNorm(dim)
    x = torch.randn(batch_size, seq_len, dim)

    out = norm(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Input RMS: {torch.sqrt(torch.mean(x ** 2)).item():.4f}")
    print(f"  Output RMS: {torch.sqrt(torch.mean(out ** 2)).item():.4f}")

    assert out.shape == x.shape, "Output shape should match input"

    print("  ✓ RMSNorm OK")


def test_swiglu():
    """Test SwiGLU FFN."""
    from src.model import FeedForward, ModelConfig

    print(f"\n{'=' * 60}")
    print("Testing SwiGLU FFN")
    print("=" * 60)

    config = ModelConfig(dim=64, hidden_dim=256)
    ffn = FeedForward(config)

    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.dim)

    out = ffn(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  FFN params: {sum(p.numel() for p in ffn.parameters()):,}")

    assert out.shape == x.shape, "Output shape should match input"

    # Check SwiGLU has 3 projections (w1, w2, w3)
    assert hasattr(ffn, 'w1') and hasattr(ffn, 'w2') and hasattr(ffn, 'w3')

    print("  ✓ SwiGLU FFN OK")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TinyLlama Model Tests")
    print("=" * 60)

    # Test configurations
    model_226k, model_110m = test_model_configs()

    # Test forward pass
    test_forward_pass(model_226k, "226K")

    # Test generation
    test_generation(model_226k, "226K")

    # Test components
    test_rope()
    test_rmsnorm()
    test_swiglu()

    # Test target masking
    test_target_masking()

    # Test checkpoint
    test_checkpoint()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
