"""
Knowledge Distillation for TinyLlama.

Trains a small student model using soft targets from a larger teacher model.

Loss = alpha * KL(student/T, teacher/T) * T^2 + (1-alpha) * CE(student, labels)

Usage:
    python src/training/distill.py --config configs/model_226k_v2.yaml
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model import TinyLlama, ModelConfig
from src.tokenizer import Tokenizer
from src.data.dataset import TextFileDataset, create_dataloader
from src.training.trainer import TrainingConfig, get_lr


class Distiller:
    """Knowledge distillation trainer."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader,
        val_loader,
        config: TrainingConfig,
        alpha: float = 0.5,
        temperature: float = 2.0,
        reverse_kld: bool = False,
        device: str = "cuda"
    ):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.alpha = alpha
        self.temperature = temperature
        self.reverse_kld = reverse_kld
        self.device = device

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Dtype setup
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }.get(config.dtype, torch.float32)

        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            self.dtype = torch.float16

        self.use_amp = self.dtype != torch.float32
        self.scaler = GradScaler('cuda', enabled=(self.dtype == torch.float16))

        # Optimizer (student only)
        decay_params = []
        no_decay_params = []
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1 or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0}
            ],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            fused=torch.cuda.is_available()
        )

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.best_val_loss = float('inf')

    def distill_loss(self, student_logits, teacher_logits, labels):
        """Compute combined distillation + CE loss.

        Supports two KL divergence modes:
        - Forward KLD (default): KL(teacher || student) - "mean-seeking"
          Student tries to cover all teacher modes. Can spread mass too thin.
        - Reverse KLD: KL(student || teacher) - "mode-seeking"
          Student focuses on main teacher modes. Better for small models
          with limited capacity (MINIPLM, ICLR 2025).
        """
        T = self.temperature

        if self.reverse_kld:
            # Reverse KLD: KL(student || teacher) - mode-seeking
            # Better for small student models (prevents overestimating low-prob regions)
            teacher_log_soft = F.log_softmax(teacher_logits / T, dim=-1)
            student_soft = F.softmax(student_logits / T, dim=-1)
            kl_loss = F.kl_div(teacher_log_soft, student_soft, reduction='batchmean') * (T * T)
        else:
            # Forward KLD: KL(teacher || student) - mean-seeking (standard)
            student_soft = F.log_softmax(student_logits / T, dim=-1)
            teacher_soft = F.softmax(teacher_logits / T, dim=-1)
            kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)

        # Standard CE on hard targets
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-1
        )

        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss, kl_loss.item(), ce_loss.item()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate student on val set."""
        self.student.eval()
        total_loss = 0.0
        count = 0

        for i, batch in enumerate(self.val_loader):
            if i >= self.config.eval_iters:
                break

            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            with autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                _, loss = self.student(input_ids, labels)

            total_loss += loss.item()
            count += 1

        self.student.train()
        return total_loss / count if count > 0 else float('inf')

    def train(self):
        """Run distillation loop."""
        t_params = sum(p.numel() for p in self.teacher.parameters())
        s_params = sum(p.numel() for p in self.student.parameters())
        print(f"Teacher: {t_params:,} params (frozen)")
        print(f"Student: {s_params:,} params (training)")
        print(f"Compression: {t_params/s_params:.1f}x")
        kld_mode = "reverse (mode-seeking)" if self.reverse_kld else "forward (mean-seeking)"
        print(f"Alpha: {self.alpha}, Temperature: {self.temperature}, KLD: {kld_mode}")
        print(f"Starting distillation from step {self.step}\n")

        self.student.train()
        train_iter = iter(self.train_loader)

        running_loss = 0.0
        running_kl = 0.0
        running_ce = 0.0
        start_time = time.time()

        while self.step < self.config.max_iters:
            lr = get_lr(self.step, self.config)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            accumulated_loss = 0.0
            accumulated_kl = 0.0
            accumulated_ce = 0.0

            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                with autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                    # Teacher forward (no grad)
                    with torch.no_grad():
                        teacher_logits, _ = self.teacher(input_ids)

                    # Student forward
                    student_logits, _ = self.student(input_ids)

                    # Distillation loss
                    loss, kl, ce = self.distill_loss(student_logits, teacher_logits, labels)
                    loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item() * self.config.gradient_accumulation_steps
                accumulated_kl += kl
                accumulated_ce += ce

            # Gradient clipping + step
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1
            running_loss += accumulated_loss
            running_kl += accumulated_kl / self.config.gradient_accumulation_steps
            running_ce += accumulated_ce / self.config.gradient_accumulation_steps

            # Logging
            if self.step % self.config.log_interval == 0:
                n = self.config.log_interval
                avg_loss = running_loss / n
                avg_kl = running_kl / n
                avg_ce = running_ce / n
                elapsed = time.time() - start_time
                tok_s = (
                    n * self.config.batch_size *
                    self.config.gradient_accumulation_steps *
                    batch['input_ids'].shape[1]
                ) / elapsed

                print(
                    f"step {self.step:6d} | "
                    f"loss {avg_loss:.4f} (kl={avg_kl:.3f} ce={avg_ce:.3f}) | "
                    f"lr {lr:.2e} | tok/s {tok_s:.0f}"
                )

                running_loss = 0.0
                running_kl = 0.0
                running_ce = 0.0
                start_time = time.time()

            # Eval
            if self.step % self.config.eval_interval == 0 and self.val_loader:
                val_loss = self.evaluate()
                print(f"step {self.step:6d} | val_loss {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
                    print(f"New best model saved (val_loss: {val_loss:.4f})")

            if self.step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.step}.pt")

        self.save_checkpoint("final.pt")
        print("Distillation complete!")

    def save_checkpoint(self, filename):
        path = self.output_dir / filename
        torch.save({
            'step': self.step,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.student.config if hasattr(self.student, 'config') else None,
        }, path)
        print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation for TinyLlama")
    parser.add_argument("--config", type=str, default="configs/model_226k_v2.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print(f"Device: {args.device}")

    distill_config = config.get('distill', {})

    # Load tokenizer
    tok_path = config['tokenizer']['model_path']
    tokenizer = Tokenizer(tok_path)
    print(f"Tokenizer: {tokenizer.vocab_size} tokens")

    # Load teacher
    teacher_path = distill_config['teacher_checkpoint']
    print(f"Loading teacher from {teacher_path}...")
    teacher = TinyLlama.from_checkpoint(teacher_path, device=args.device)
    teacher.eval()
    print(f"Teacher: {sum(p.numel() for p in teacher.parameters()):,} params")

    # Create student
    model_cfg = ModelConfig(**config['model'])
    model_cfg.vocab_size = tokenizer.vocab_size
    student = TinyLlama(model_cfg)
    print(f"Student: {student.count_parameters():,} params")

    # Load data
    data_path = distill_config.get('data_path', 'data/tinystories')
    batch_size = distill_config.get('batch_size', 64)

    train_dataset = TextFileDataset(data_path, tokenizer, model_cfg.max_seq_len, split="train")
    val_dataset = TextFileDataset(data_path, tokenizer, model_cfg.max_seq_len, split="validation")

    train_loader = create_dataloader(train_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = create_dataloader(val_dataset, batch_size, shuffle=False, num_workers=0, drop_last=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Training config
    output_dir = config.get('output_dir', 'checkpoints/226k_v2')
    train_config = TrainingConfig(
        learning_rate=distill_config.get('learning_rate', 1e-3),
        min_lr=distill_config.get('min_lr', 1e-4),
        warmup_iters=distill_config.get('warmup_iters', 300),
        max_iters=distill_config.get('max_iters', 10000),
        weight_decay=distill_config.get('weight_decay', 0.1),
        beta1=distill_config.get('beta1', 0.9),
        beta2=distill_config.get('beta2', 0.95),
        grad_clip=distill_config.get('grad_clip', 1.0),
        lr_schedule=distill_config.get('lr_schedule', 'cosine'),
        stable_frac=distill_config.get('stable_frac', 0.7),
        decay_frac=distill_config.get('decay_frac', 0.2),
        batch_size=batch_size,
        gradient_accumulation_steps=distill_config.get('gradient_accumulation_steps', 2),
        log_interval=distill_config.get('log_interval', 10),
        eval_interval=distill_config.get('eval_interval', 500),
        eval_iters=distill_config.get('eval_iters', 100),
        save_interval=distill_config.get('save_interval', 2000),
        dtype=distill_config.get('dtype', 'bfloat16'),
        compile=distill_config.get('compile', False),
        output_dir=os.path.join(output_dir, 'distill')
    )

    # Run distillation
    distiller = Distiller(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        alpha=distill_config.get('alpha', 0.5),
        temperature=distill_config.get('temperature', 2.0),
        reverse_kld=distill_config.get('reverse_kld', False),
        device=args.device
    )

    print("\n" + "=" * 50)
    print("Starting knowledge distillation...")
    print("=" * 50 + "\n")

    distiller.train()

    # Final eval
    val_loss = distiller.evaluate()
    print(f"\nFinal val_loss: {val_loss:.4f}")
    print(f"Final perplexity: {math.exp(val_loss):.2f}")


if __name__ == "__main__":
    main()
