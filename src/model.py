"""
TinyLlama Model Architecture

Implements Llama 2 style architecture with:
- RoPE (Rotary Position Embeddings)
- RMSNorm (instead of LayerNorm)
- SwiGLU activation in FFN
- No bias in linear layers
- Grouped Query Attention (GQA) support
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

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

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config['model'])

    def param_count(self) -> int:
        """Estimate parameter count."""
        # Embedding
        embed = self.vocab_size * self.dim

        # Per layer
        head_dim = self.dim // self.n_heads
        # Attention: Q, K, V projections + output
        attn_q = self.dim * self.dim  # Q projection
        attn_kv = 2 * self.dim * (self.n_kv_heads * head_dim)  # K, V projections
        attn_o = self.dim * self.dim  # Output projection
        attn = attn_q + attn_kv + attn_o

        # FFN: SwiGLU has 3 projections
        ffn = 3 * self.dim * self.hidden_dim

        # Norms: 2 per layer + 1 final
        norms_per_layer = 2 * self.dim

        layer_params = attn + ffn + norms_per_layer
        all_layers = layer_params * self.n_layers

        # Final norm
        final_norm = self.dim

        # Output projection (may be tied with embedding)
        output = 0 if self.weight_tying else self.vocab_size * self.dim

        return embed + all_layers + final_norm + output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute rotary embedding frequencies."""
    # Compute frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Compute positions
    t = torch.arange(max_seq_len)
    # Outer product: [max_seq_len, dim/2]
    freqs = torch.outer(t, freqs)
    # Complex exponential: e^(i * theta) = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys."""
    # Reshape to complex: [..., dim] -> [..., dim/2, 2] -> [..., dim/2] (complex)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs for broadcasting: [seq_len, dim/2] -> [1, seq_len, 1, dim/2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) support."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # How many times to repeat KV

        # Projections (no bias, Llama style)
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        xq = self.wq(x)  # [B, T, n_heads * head_dim]
        xk = self.wk(x)  # [B, T, n_kv_heads * head_dim]
        xv = self.wv(x)  # [B, T, n_kv_heads * head_dim]

        # Reshape for multi-head attention
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Repeat K, V for GQA
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Transpose for attention: [B, n_heads, T, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        output = torch.matmul(attn, xv)  # [B, n_heads, T, head_dim]

        # Reshape back: [B, T, dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # SwiGLU: gate * swish(x)
        # We need 3 projections: w1 (gate), w2 (down), w3 (up)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        # Pre-norm FFN with residual
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TinyLlama(nn.Module):
    """TinyLlama language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
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

        # Precompute RoPE frequencies
        head_dim = config.dim // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len),
            persistent=False
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            tokens: Input token ids [B, T]
            targets: Target token ids for loss computation [B, T]
            target_mask: Mask for selective loss (1 for positions to include)

        Returns:
            logits: Output logits [B, T, vocab_size]
            loss: Optional cross-entropy loss
        """
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.config.max_seq_len, \
            f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}"

        # Token embeddings
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:seq_len]

        # Causal mask
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=tokens.device),
            diagonal=1
        )

        # Apply transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        # Final norm and output projection
        h = self.norm(h)
        logits = self.output(h)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Shift logits and targets for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()

            # Flatten for cross-entropy
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_targets = shift_targets.view(-1)

            if target_mask is not None:
                # Apply mask for selective loss (e.g., only assistant tokens)
                shift_mask = target_mask[..., 1:].contiguous().view(-1)
                # Compute per-token loss
                loss = F.cross_entropy(shift_logits, shift_targets, reduction='none')
                # Apply mask and compute mean over valid tokens
                loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
            else:
                loss = F.cross_entropy(shift_logits, shift_targets)

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
        """
        Generate tokens autoregressively.

        Args:
            tokens: Input token ids [B, T] or [T]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            stop_tokens: List of token ids that stop generation

        Returns:
            Generated token ids including input tokens
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        stop_tokens = stop_tokens or []

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            tokens_cond = tokens if tokens.size(1) <= self.config.max_seq_len \
                else tokens[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self.forward(tokens_cond)
            logits = logits[:, -1, :]  # Last position

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Append token
            tokens = torch.cat([tokens, next_token], dim=1)

            # Check stop tokens
            if next_token.item() in stop_tokens:
                break

        return tokens

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
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def test_model():
    """Test model instantiation and forward pass."""
    print("Testing TinyLlama model...")

    # Test 226K config
    config_226k = ModelConfig(
        dim=48, n_layers=7, n_heads=4, n_kv_heads=2,
        vocab_size=1024, hidden_dim=128, max_seq_len=128,
        dropout=0.15, weight_tying=True
    )

    model_226k = TinyLlama(config_226k)
    params_226k = model_226k.count_parameters()
    print(f"226K model parameters: {params_226k:,}")
    print(f"  Estimated: {config_226k.param_count():,}")

    # Test forward pass
    batch = torch.randint(0, 1024, (2, 32))
    logits, loss = model_226k(batch, batch)
    print(f"  Forward pass OK: logits {logits.shape}, loss {loss.item():.4f}")

    # Test generation
    gen = model_226k.generate(batch[0, :5], max_new_tokens=10)
    print(f"  Generation OK: {gen.shape}")

    # Test 110M config
    config_110m = ModelConfig(
        dim=768, n_layers=12, n_heads=12, n_kv_heads=12,
        vocab_size=32000, hidden_dim=2048, max_seq_len=1024,
        dropout=0.0, weight_tying=True
    )

    model_110m = TinyLlama(config_110m)
    params_110m = model_110m.count_parameters()
    print(f"\n110M model parameters: {params_110m:,}")
    print(f"  Estimated: {config_110m.param_count():,}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_model()
