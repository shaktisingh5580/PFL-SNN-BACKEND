"""
Original training entry point — fixed interface.
Prefer using scripts/train_backend.py which is the CPU-optimized version.
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from config.settings import (
    OSCD_DIR, MODELS_DIR, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    BATCH_SIZE, GRADIENT_CLIP, BETA, NUM_STEPS, NUM_BANDS, ENCODER_CHANNELS,
    LOSS_ALPHA, CLASS_WEIGHT_CHANGE,
)


def main():
    from src.model.siamese_snn import SiameseSNN
    from src.model.losses import CombinedLoss
    from src.training.trainer import SiameseSNNTrainer   # ← correct class name
    from src.training.dataset import OSCDDataset
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    # Dataset
    try:
        train_dataset = OSCDDataset(str(OSCD_DIR), split="train")
        test_dataset  = OSCDDataset(str(OSCD_DIR), split="test")
        logger.info(f"OSCD loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    except Exception as e:
        logger.warning(f"OSCD not available: {e}. Using synthetic demo data.")
        from torch.utils.data import TensorDataset
        n = 20
        train_dataset = TensorDataset(
            torch.randn(n, NUM_BANDS, 256, 256),
            torch.randn(n, NUM_BANDS, 256, 256),
            torch.randint(0, 2, (n, 256, 256)),
        )
        test_dataset = TensorDataset(
            torch.randn(5, NUM_BANDS, 256, 256),
            torch.randn(5, NUM_BANDS, 256, 256),
            torch.randint(0, 2, (5, 256, 256)),
        )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = SiameseSNN(
        in_channels=NUM_BANDS,
        encoder_channels=ENCODER_CHANNELS,
        beta=BETA,
        num_steps=NUM_STEPS,
    )
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss
    loss_fn = CombinedLoss(alpha=LOSS_ALPHA, change_weight=CLASS_WEIGHT_CHANGE)

    # Train
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trainer = SiameseSNNTrainer(              # ← fixed class name
        model=model,
        loss_fn=loss_fn,                       # ← required arg now passed
        train_loader=train_loader,
        val_loader=test_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_epochs=NUM_EPOCHS,
        gradient_clip=GRADIENT_CLIP,
        device=device,
        checkpoint_dir=str(MODELS_DIR),
    )

    trainer.train()                            # ← no args; num_epochs set in __init__
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
