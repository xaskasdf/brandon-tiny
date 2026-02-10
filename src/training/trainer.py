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
    Compute learning rate with warmup and cosine decay.

    Args:
        step: Current training step
        config: Training configuration

    Returns:
        Learning rate for this step
    """
    # Linear warmup
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / config.warmup_iters

    # Cosine decay
    if step >= config.max_iters:
        return config.min_lr

    decay_ratio = (step - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


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

    def train_step(self, batch: dict) -> float:
        """
        Perform a single training step.

        Args:
            batch: Batch of training data

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
            _, loss = self.model(input_ids, labels, target_mask)
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

        while self.step < self.config.max_iters:
            # Update learning rate
            lr = get_lr(self.step, self.config)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Accumulate gradients
            accumulated_loss = 0.0
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                loss = self.train_step(batch)
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

                print(
                    f"step {self.step:6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"tok/s {tokens_per_sec:.0f}"
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
