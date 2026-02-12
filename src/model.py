"""
TinyLlama Model Architecture v2

Optimized based on MobileLLM, SmolLM2, and Phi research:
- Deep-narrow architecture (more layers, smaller dim)
- Flash Attention via F.scaled_dot_product_attention
- Better weight initialization (1/sqrt(fan_in))
- Block-wise weight sharing (MobileLLM)
- KV cache for fast generation
- RoPE with configurable theta
- SwiGLU activation, RMSNorm, GQA
- Multi-Token Prediction (MTP) - predict next k tokens simultaneously
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Model configuration."""
    dim: int = 48
    n_layers: int = 7
    n_heads: int = 4
    n_kv_heads: int = 2  # For GQA; if equal to n_heads, standard MHA
    vocab_size: int = 1024
    hidden_dim: int = 128  # FFN hidden dimension
    max_seq_len: int = 128
    dropout: float = 0.15
    weight_tying: bool = True
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    block_sharing: bool = False  # MobileLLM block-wise weight sharing
    n_predict: int = 1  # Multi-Token Prediction: predict next k tokens (1=standard)
    dense_former: bool = False  # DenseFormer DWA (weighted averaging of all layer outputs)
    value_residual: bool = False  # Value Residual Learning (add layer 0's V to all layers)
    n_registers: int = 0  # Register tokens (learnable tokens prepended to sequence)

    def __post_init__(self):
        """Ensure proper types after initialization."""
        self.dim = int(self.dim)
        self.n_layers = int(self.n_layers)
        self.n_heads = int(self.n_heads)
        self.n_kv_heads = int(self.n_kv_heads)
        self.vocab_size = int(self.vocab_size)
        self.hidden_dim = int(self.hidden_dim)
        self.max_seq_len = int(self.max_seq_len)
        self.dropout = float(self.dropout)
        self.weight_tying = bool(self.weight_tying)
        self.norm_eps = float(self.norm_eps)
        self.rope_theta = float(self.rope_theta)
        self.block_sharing = bool(self.block_sharing)
        self.n_predict = int(self.n_predict)
        self.dense_former = bool(self.dense_former)
        self.value_residual = bool(self.value_residual)
        self.n_registers = int(self.n_registers)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config['model'])

    def param_count(self) -> int:
        """Estimate parameter count."""
        embed = self.vocab_size * self.dim
        head_dim = self.dim // self.n_heads

        attn_q = self.dim * self.dim
        attn_kv = 2 * self.dim * (self.n_kv_heads * head_dim)
        attn_o = self.dim * self.dim
        attn = attn_q + attn_kv + attn_o

        ffn = 3 * self.dim * self.hidden_dim
        norms_per_layer = 2 * self.dim

        layer_params = attn + ffn + norms_per_layer

        # Block sharing: only half the layers have unique params
        if self.block_sharing:
            unique_layers = (self.n_layers + 1) // 2
        else:
            unique_layers = self.n_layers
        all_layers = layer_params * unique_layers

        final_norm = self.dim
        output = 0 if self.weight_tying else self.vocab_size * self.dim

        return embed + all_layers + final_norm + output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute rotary embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys."""
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention with GQA + Flash Attention + KV cache."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout_p = config.dropout

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        v_first: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Value Residual Learning: store raw V, add layer 0's V as residual
        v_out = xv
        if v_first is not None:
            xv = xv + v_first

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Transpose for attention: [B, n_heads, T, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # KV cache for generation
        new_cache = None
        if cache is not None:
            cached_k, cached_v = cache
            xk = torch.cat([cached_k, xk], dim=2)
            xv = torch.cat([cached_v, xv], dim=2)
            new_cache = (xk, xv)
        elif not self.training:
            new_cache = (xk, xv)

        # Repeat KV for GQA
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=1)
            xv = xv.repeat_interleave(self.n_rep, dim=1)

        # Flash Attention via PyTorch SDPA
        dropout_p = self.dropout_p if self.training else 0.0

        if cache is not None:
            # Generation mode: single query token, no causal mask needed
            output = F.scaled_dot_product_attention(
                xq, xk, xv, dropout_p=dropout_p, is_causal=False
            )
        elif mask is not None:
            # Training with explicit mask
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=mask, dropout_p=dropout_p
            )
        else:
            # Training with built-in causal mask (most efficient)
            output = F.scaled_dot_product_attention(
                xq, xk, xv, dropout_p=dropout_p, is_causal=True
            )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output), new_cache, v_out


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        v_first: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        attn_out, new_cache, v_out = self.attention(self.attention_norm(x), freqs_cis, mask, cache, v_first)
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_cache, v_out


class TinyLlama(nn.Module):
    """TinyLlama language model v2."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        if config.block_sharing:
            # MobileLLM: share weights between adjacent layer pairs
            unique_layers = nn.ModuleList([
                TransformerBlock(config) for _ in range((config.n_layers + 1) // 2)
            ])
            self.layers = nn.ModuleList()
            for i in range(config.n_layers):
                self.layers.append(unique_layers[i // 2])
            # Keep unique_layers alive for parameter counting
            self._unique_layers = unique_layers
        else:
            self.layers = nn.ModuleList([
                TransformerBlock(config) for _ in range(config.n_layers)
            ])

        # Final norm
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # Output projection
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        if config.weight_tying:
            self.output.weight = self.tok_embeddings.weight

        # DenseFormer DWA: learnable weights for cross-layer averaging
        if config.dense_former:
            # After layer i, combine h_0..h_{i+1} with learned weights
            # Shape: [n_layers, n_layers + 1], only lower-triangular used
            self.dwa_weights = nn.Parameter(torch.zeros(config.n_layers, config.n_layers + 1))
            # Init: weight 1.0 on latest output = identity at start
            for i in range(config.n_layers):
                self.dwa_weights.data[i, i + 1] = 1.0

        # Register tokens: learnable tokens prepended to sequence
        if config.n_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, config.n_registers, config.dim) * 0.02
            )

        # Multi-Token Prediction: extra heads for predicting t+2, t+3, ..., t+k
        # Each head is a dim->dim transform; all share self.output for final projection
        if config.n_predict > 1:
            self.mtp_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.dim, config.dim, bias=False),
                    RMSNorm(config.dim, config.norm_eps),
                )
                for _ in range(config.n_predict - 1)
            ])

        # Precompute RoPE frequencies
        head_dim = config.dim // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len * 2, config.rope_theta),
            persistent=False
        )

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers)
        self._init_residual_scaling()

    def _init_weights(self, module: nn.Module):
        """Initialize weights with 1/sqrt(fan_in) scaling."""
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            std = 1.0 / math.sqrt(fan_in)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 1.0 / math.sqrt(module.weight.shape[1])
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def _init_residual_scaling(self):
        """Scale residual output projections to prevent growth."""
        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        for layer in self.layers:
            # Scale attention output projection
            layer.attention.wo.weight.data *= scale
            # Scale FFN output projection
            layer.feed_forward.w2.weight.data *= scale

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        n_predict_override: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for training.

        Args:
            tokens: Input token IDs [B, T]
            targets: Target token IDs [B, T]
            target_mask: Mask for target-only loss (fine-tuning)
            n_predict_override: Override number of MTP heads to use (for curriculum)
        """
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.config.max_seq_len, \
            f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}"

        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # Register tokens: prepend learnable tokens
        n_reg = self.config.n_registers
        if n_reg > 0:
            regs = self.register_tokens.expand(batch_size, -1, -1)
            h = torch.cat([regs, h], dim=1)

        freqs_cis = self.freqs_cis[:seq_len + n_reg]

        # Track layer outputs for DenseFormer DWA
        all_h = [h] if self.config.dense_former else None
        v_first = None

        for i, layer in enumerate(self.layers):
            vf = v_first if self.config.value_residual else None
            h, _, v_out = layer(h, freqs_cis, mask=None, cache=None, v_first=vf)

            # Value Residual: store layer 0's V for subsequent layers
            if i == 0 and self.config.value_residual:
                v_first = v_out

            # DenseFormer DWA: weighted average of all layer outputs
            if self.config.dense_former:
                all_h.append(h)
                w = self.dwa_weights[i, :i + 2]
                h = sum(w[j] * all_h[j] for j in range(i + 2))

        # Strip register tokens before output
        if n_reg > 0:
            h = h[:, n_reg:]

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            V = self.config.vocab_size
            n_predict_active = n_predict_override if n_predict_override is not None else self.config.n_predict

            if target_mask is not None or n_predict_active <= 1:
                # Standard next-token loss (fine-tuning or MTP disabled/curriculum k=1)
                shift_logits = logits[..., :-1, :].contiguous().view(-1, V)
                shift_targets = targets[..., 1:].contiguous().view(-1)

                if target_mask is not None:
                    shift_mask = target_mask[..., 1:].contiguous().view(-1)
                    loss = F.cross_entropy(shift_logits, shift_targets, reduction='none')
                    loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
                else:
                    loss = F.cross_entropy(shift_logits, shift_targets)
            else:
                # Multi-Token Prediction: average loss over k prediction heads
                total_loss = torch.zeros(1, device=tokens.device)
                n_heads_used = 0

                # Head 0: standard next-token (shift=1)
                shift_logits = logits[..., :-1, :].contiguous().view(-1, V)
                shift_targets = targets[..., 1:].contiguous().view(-1)
                total_loss += F.cross_entropy(shift_logits, shift_targets)
                n_heads_used += 1

                # Extra heads: predict t+2, ..., t+k (limited by curriculum)
                n_extra = min(n_predict_active - 1, len(self.mtp_heads))
                for i in range(n_extra):
                    head = self.mtp_heads[i]
                    shift = i + 2  # head 0=shift 1, extra heads start at shift 2
                    if shift >= seq_len:
                        break
                    head_logits = self.output(head(h))
                    s_logits = head_logits[..., :-shift, :].contiguous().view(-1, V)
                    s_targets = targets[..., shift:].contiguous().view(-1)
                    total_loss += F.cross_entropy(s_logits, s_targets)
                    n_heads_used += 1

                loss = total_loss / n_heads_used

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        stop_tokens: Optional[list] = None
    ) -> torch.Tensor:
        """Generate tokens with KV cache for speed."""
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        stop_tokens = stop_tokens or []
        self.eval()

        # Prefill: process all input tokens at once
        batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        # Register tokens: prepend
        n_reg = self.config.n_registers
        if n_reg > 0:
            regs = self.register_tokens.expand(batch_size, -1, -1)
            h = torch.cat([regs, h], dim=1)

        freqs_cis = self.freqs_cis[:seq_len + n_reg]

        caches = [None] * len(self.layers)
        all_h = [h] if self.config.dense_former else None
        v_first = None

        for i, layer in enumerate(self.layers):
            vf = v_first if self.config.value_residual else None
            h, caches[i], v_out = layer(h, freqs_cis, mask=None, cache=None, v_first=vf)
            if i == 0 and self.config.value_residual:
                v_first = v_out
            if self.config.dense_former:
                all_h.append(h)
                w = self.dwa_weights[i, :i + 2]
                h = sum(w[j] * all_h[j] for j in range(i + 2))

        # Strip registers for logits
        if n_reg > 0:
            h = h[:, n_reg:]

        h = self.norm(h)
        logits = self.output(h[:, -1:, :])  # Only last position

        # Sample first token
        generated = tokens
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :]

            if temperature > 0:
                next_logits = next_logits / temperature

                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() in stop_tokens:
                break

            if step < max_new_tokens - 1:
                # Decode step: only process the new token with KV cache
                pos = seq_len + n_reg + step
                if pos >= self.config.max_seq_len + n_reg:
                    break

                h = self.tok_embeddings(next_token)
                freqs_cis_step = self.freqs_cis[pos:pos+1]

                all_h = [h] if self.config.dense_former else None
                v_first_step = None

                for i, layer in enumerate(self.layers):
                    vf = v_first_step if self.config.value_residual else None
                    h, caches[i], v_out = layer(h, freqs_cis_step, mask=None, cache=caches[i], v_first=vf)
                    if i == 0 and self.config.value_residual:
                        v_first_step = v_out
                    if self.config.dense_former:
                        all_h.append(h)
                        w = self.dwa_weights[i, :i + 2]
                        h = sum(w[j] * all_h[j] for j in range(i + 2))

                h = self.norm(h)
                logits = self.output(h)

        return generated

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str, optimizer=None, step: int = 0, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'step': step,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cuda') -> "TinyLlama":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']
        # Handle missing config fields from older checkpoints
        for attr, default in [('dense_former', False), ('value_residual', False),
                              ('n_registers', 0), ('n_predict', 1)]:
            if not hasattr(config, attr):
                setattr(config, attr, default)
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)


def test_model():
    """Test model instantiation and forward pass."""
    print("Testing TinyLlama v2 model...")

    # Test deep-narrow 10M config
    config_10m = ModelConfig(
        dim=192, n_layers=16, n_heads=6, n_kv_heads=2,
        vocab_size=8192, hidden_dim=512, max_seq_len=512,
        dropout=0.05, weight_tying=True, block_sharing=True
    )

    model = TinyLlama(config_10m)
    params = model.count_parameters()
    print(f"10M v2 model parameters: {params:,}")
    print(f"  Estimated: {config_10m.param_count():,}")

    # Test forward
    batch = torch.randint(0, 8192, (2, 64))
    logits, loss = model(batch, batch)
    print(f"  Forward pass OK: logits {logits.shape}, loss {loss.item():.4f}")

    # Test generation with KV cache
    gen = model.generate(batch[0, :5], max_new_tokens=20, temperature=0.8)
    print(f"  Generation OK (KV cache): {gen.shape}")

    # Test 30M deep-narrow
    config_30m = ModelConfig(
        dim=320, n_layers=20, n_heads=8, n_kv_heads=2,
        vocab_size=8192, hidden_dim=864, max_seq_len=512,
        dropout=0.0, weight_tying=True, block_sharing=True
    )

    model_30m = TinyLlama(config_30m)
    print(f"\n30M v2 model parameters: {model_30m.count_parameters():,}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_model()
