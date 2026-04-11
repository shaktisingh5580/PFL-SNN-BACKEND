"""
Training loop implementing BPTT over T time-steps for the Siamese-SNN.
"""
import logging
import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


class SiameseSNNTrainer:
    """
    Trainer for the hybrid Siamese-SNN model.
    Implements BPTT over T=20 time-steps with mixed precision training.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        gradient_clip: float = 1.0,
        device: str = "auto",
        checkpoint_dir: str = "outputs/models",
        use_amp: bool = True,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )

        # Mixed precision
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Tracking
        self.best_f1 = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_f1": []}

        logger.info(f"Trainer initialized on {self.device} (AMP: {self.use_amp})")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (img_a, img_b, mask) in enumerate(self.train_loader):
            img_a = img_a.to(self.device)
            img_b = img_b.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    spk_rec, mem_rec = self.model(img_a, img_b)
                    loss = self.loss_fn(spk_rec, mask)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                spk_rec, mem_rec = self.model(img_a, img_b)
                loss = self.loss_fn(spk_rec, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute metrics."""
        if self.val_loader is None:
            return {"val_loss": 0.0, "f1": 0.0, "iou": 0.0}

        self.model.eval()
        total_loss = 0.0
        tp, fp, fn, tn = 0, 0, 0, 0
        num_batches = 0

        for img_a, img_b, mask in self.val_loader:
            img_a = img_a.to(self.device)
            img_b = img_b.to(self.device)
            mask = mask.to(self.device)

            spk_rec, mem_rec = self.model(img_a, img_b)
            loss = self.loss_fn(spk_rec, mask)
            total_loss += loss.item()

            # Get predictions
            pred = self.model.predict(img_a, img_b)

            # Compute confusion matrix elements
            tp += ((pred == 1) & (mask == 1)).sum().item()
            fp += ((pred == 1) & (mask == 0)).sum().item()
            fn += ((pred == 0) & (mask == 1)).sum().item()
            tn += ((pred == 0) & (mask == 0)).sum().item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Compute metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

        return {
            "val_loss": avg_loss,
            "f1": f1,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }

    def train(self) -> Dict:
        """Full training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_metrics = self.validate()
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_f1"].append(val_metrics["f1"])

            # Learning rate step
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model
            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  ★ New best F1: {self.best_f1:.4f}")

            # Periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time/60:.1f} minutes. Best F1: {self.best_f1:.4f}")

        return self.history

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_f1": self.best_f1,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, str(path))
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_f1 = checkpoint.get("best_f1", 0.0)
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
