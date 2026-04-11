"""
MAE self-supervised pre-training loop.
Trains on unlabeled Sentinel-2 tiles with 75% masking.
"""
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

logger = logging.getLogger(__name__)


def pretrain_mae(
    model,
    dataloader: DataLoader,
    num_epochs: int = 300,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 40,
    device: str = "auto",
    checkpoint_dir: str = "outputs/models/mae",
    log_interval: int = 10,
):
    """
    Self-supervised MAE pre-training loop.

    Args:
        model: MaskedAutoencoder instance
        dataloader: DataLoader of unlabeled tiles
        num_epochs: Total training epochs
        lr: Peak learning rate
        weight_decay: AdamW weight decay
        warmup_epochs: Linear warmup epochs
        device: Compute device
        checkpoint_dir: Directory for saving checkpoints
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    # Cosine schedule with warmup
    total_steps = num_epochs * len(dataloader)
    warmup_steps = warmup_epochs * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info(f"MAE pre-training: {num_epochs} epochs, {len(dataloader)} batches/epoch, device={device}")
    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start = time.time()

        for batch_idx, tiles in enumerate(dataloader):
            if isinstance(tiles, (list, tuple)):
                tiles = tiles[0]
            tiles = tiles.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss, pred, mask = model(tiles)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, pred, mask = model(tiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if batch_idx % log_interval == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.6f} LR: {scheduler.get_last_lr()[0]:.6f}"
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s"
        )

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, str(ckpt_dir / "mae_best.pt"))

        # Periodic checkpoint
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
            }, str(ckpt_dir / f"mae_epoch_{epoch}.pt"))

    # Save encoder weights only (for transfer learning)
    torch.save(
        model.encoder.state_dict(),
        str(ckpt_dir / "mae_encoder_weights.pt"),
    )
    logger.info(f"Pre-training complete. Best loss: {best_loss:.6f}")
    logger.info(f"Encoder weights saved to {ckpt_dir / 'mae_encoder_weights.pt'}")
