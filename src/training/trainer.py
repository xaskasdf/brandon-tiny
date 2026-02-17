"""
Training utilities for TinyLlama.

Implements:
- Cosine learning rate schedule with warmup
- Gradient clipping
- Mixed precision training
- Checkpointing
- Logging (console + optional wandb)
"""

import os
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 500
    max_iters: int = 10000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule
    lr_schedule: str = "cosine"  # "cosine" or "wsd" (Warmup-Stable-Decay)
    stable_frac: float = 0.7    # WSD: fraction of training at peak LR
    decay_frac: float = 0.2     # WSD: fraction for cosine decay

    # Batch
    batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 100
    save_interval: int = 1000

    # Mixed precision
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"

    # MTP curriculum
    mtp_curriculum: bool = False  # Enable gradual MTP ramp-up
    mtp_warmup_frac: float = 0.4  # Fraction of training to ramp from k=1 to k=n_predict

    # Anti-repetition training
    label_smoothing: float = 0.0  # Label smoothing epsilon (0.1 recommended)
    unlikelihood_alpha: float = 0.0  # Unlikelihood training weight (0.5 recommended)
    entropy_reg_beta: float = 0.0  # Entropy regularization weight (0.01 recommended)

    # Misc
    compile: bool = False
    output_dir: str = "checkpoints"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str, section: str = "pretrain") -> "TrainingConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        train_config = config.get(section, {})
        # Also grab output_dir from root
        train_config['output_dir'] = config.get('output_dir', 'checkpoints')
        return cls(**train_config)


def get_lr(step: int, config: TrainingConfig) -> float:
    """
    Compute learning rate with warmup and cosine decay or WSD schedule.

    WSD (Warmup-Stable-Decay) from MiniCPM:
      - Warmup: linear ramp to peak LR
      - Stable: constant at peak LR (majority of training)
      - Decay: cosine decay to min_lr

    Args:
        step: Current training step
        config: Training configuration

    Returns:
        Learning rate for this step
    """
    # Linear warmup (same for both schedules)
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / config.warmup_iters

    if step >= config.max_iters:
        return config.min_lr

    if config.lr_schedule == "wsd":
        # WSD: Warmup-Stable-Decay
        remaining = config.max_iters - config.warmup_iters
        stable_end = config.warmup_iters + int(remaining * config.stable_frac)
        decay_steps = int(remaining * config.decay_frac)
        decay_start = config.max_iters - decay_steps

        if step < stable_end:
            # Stable phase: constant peak LR
            return config.learning_rate
        elif step < decay_start:
            # Transition: still at peak (gap between stable and decay)
            return config.learning_rate
        else:
            # Decay phase: cosine decay
            decay_ratio = (step - decay_start) / max(1, config.max_iters - decay_start)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    else:
        # Cosine decay (default)
        decay_ratio = (step - config.warmup_iters) / (config.max_iters - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def get_mtp_k(step: int, config: TrainingConfig, n_predict: int) -> int:
    """
    Compute number of active MTP heads using forward curriculum.

    Small models struggle with full MTP from the start. This gradually increases
    from k=1 (standard NTP) to k=n_predict over the warmup fraction of training.

    Schedule for n_predict=4, mtp_warmup_frac=0.4, max_iters=15000:
      Steps 0-2000:   k=1 (pure NTP)
      Steps 2000-4000: k=2
      Steps 4000-6000: k=3
      Steps 6000+:     k=4 (full MTP)

    Args:
        step: Current training step
        config: Training configuration
        n_predict: Maximum number of tokens to predict (from model config)

    Returns:
        Number of active prediction heads (1 = NTP only)
    """
    if not config.mtp_curriculum or n_predict <= 1:
        return n_predict

    warmup_steps = int(config.max_iters * config.mtp_warmup_frac)
    if step >= warmup_steps:
        return n_predict

    # Each phase adds one more prediction head
    phase_len = warmup_steps / (n_predict - 1) if n_predict > 1 else 1
    k = 1 + int(step / phase_len)
    return min(k, n_predict)


class Trainer:
    """
    Trainer class for TinyLlama.

    Handles the training loop with all optimizations.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        device: str = "cuda"
    ):
        """
        Initialize trainer.

        Args:
            model: TinyLlama model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        # Setup dtype
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }.get(config.dtype, torch.float32)

        # Check bfloat16 support
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("Warning: bfloat16 not supported, falling back to float16")
            self.dtype = torch.float16

        # Setup mixed precision
        self.use_amp = self.dtype != torch.float32
        self.scaler = GradScaler('cuda', enabled=(self.dtype == torch.float16))

        # Compile model if requested
        if config.compile and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.step = 0
        self.best_val_loss = float('inf')

        # Wandb
        self.wandb_run = None
        if config.wandb_project:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=vars(config)
                )
            except ImportError:
                print("Warning: wandb not installed, skipping logging")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters into those with and without weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and norms
            if param.dim() == 1 or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            fused=torch.cuda.is_available()  # Use fused optimizer if available
        )

        return optimizer

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate model on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        count = 0

        for i, batch in enumerate(self.val_loader):
            if i >= self.config.eval_iters:
                break

            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            target_mask = batch.get('target_mask')
            if target_mask is not None:
                target_mask = target_mask.to(self.device)

            # Forward pass with mixed precision
            with autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                _, loss = self.model(input_ids, labels, target_mask)

            total_loss += loss.item()
            count += 1

        self.model.train()
        return total_loss / count if count > 0 else float('inf')

    def train_step(self, batch: dict, mtp_k: Optional[int] = None) -> float:
        """
        Perform a single training step.

        Args:
            batch: Batch of training data
            mtp_k: Number of active MTP heads (curriculum override)

        Returns:
            Loss value
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        target_mask = batch.get('target_mask')
        if target_mask is not None:
            target_mask = target_mask.to(self.device)

        # Forward pass with mixed precision
        with autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
            _, loss = self.model(
                input_ids, labels, target_mask,
                n_predict_override=mtp_k,
                label_smoothing=self.config.label_smoothing,
                unlikelihood_alpha=self.config.unlikelihood_alpha,
                entropy_reg_beta=self.config.entropy_reg_beta,
            )
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def train(self, resume_from: Optional[str] = None):
        """
        Run training loop.

        Args:
            resume_from: Optional path to checkpoint to resume from
        """
        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        print(f"Starting training from step {self.step}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.model.train()
        train_iter = iter(self.train_loader)

        # Training metrics
        running_loss = 0.0
        start_time = time.time()

        # MTP curriculum setup
        n_predict = getattr(self.model, 'config', None)
        n_predict = n_predict.n_predict if n_predict and hasattr(n_predict, 'n_predict') else 1
        if self.config.mtp_curriculum and n_predict > 1:
            print(f"MTP curriculum enabled: ramping k=1→{n_predict} over "
                  f"{int(self.config.max_iters * self.config.mtp_warmup_frac)} steps")

        while self.step < self.config.max_iters:
            # Update learning rate
            lr = get_lr(self.step, self.config)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Compute MTP curriculum k
            mtp_k = get_mtp_k(self.step, self.config, n_predict)

            # Accumulate gradients
            accumulated_loss = 0.0
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                loss = self.train_step(batch, mtp_k=mtp_k)
                accumulated_loss += loss

            # Gradient clipping
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1
            running_loss += accumulated_loss

            # Logging
            if self.step % self.config.log_interval == 0:
                avg_loss = running_loss / self.config.log_interval
                elapsed = time.time() - start_time
                tokens_per_sec = (
                    self.config.log_interval *
                    self.config.batch_size *
                    self.config.gradient_accumulation_steps *
                    batch['input_ids'].shape[1]
                ) / elapsed

                mtp_str = f" | mtp_k={mtp_k}" if n_predict > 1 else ""
                print(
                    f"step {self.step:6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"tok/s {tokens_per_sec:.0f}"
                    f"{mtp_str}"
                )

                if self.wandb_run:
                    import wandb
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lr': lr,
                        'train/tokens_per_sec': tokens_per_sec
                    }, step=self.step)

                running_loss = 0.0
                start_time = time.time()

            # Evaluation
            if self.step % self.config.eval_interval == 0 and self.val_loader:
                val_loss = self.evaluate()
                print(f"step {self.step:6d} | val_loss {val_loss:.4f}")

                if self.wandb_run:
                    import wandb
                    wandb.log({'val/loss': val_loss}, step=self.step)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
                    print(f"New best model saved (val_loss: {val_loss:.4f})")

            # Regular checkpoint
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.step}.pt")

        # Final save
        self.save_checkpoint("final.pt")
        print("Training complete!")

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.output_dir / filename
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.config if hasattr(self.model, 'config') else None
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from step {self.step}")


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    num_batches: int = 100,
    device: str = "cuda"
) -> float:
    """
    Estimate loss over multiple batches.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        num_batches: Number of batches to evaluate
        device: Device

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    count = 0

    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        target_mask = batch.get('target_mask')
        if target_mask is not None:
            target_mask = target_mask.to(device)

        _, loss = model(input_ids, labels, target_mask)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / count if count > 0 else float('inf')


if __name__ == "__main__":
    print("Trainer module loaded successfully")
