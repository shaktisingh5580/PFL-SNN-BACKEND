"""Launch MAE self-supervised pre-training on Surat tiles."""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    RAW_DIR, MODELS_DIR, MAE_EPOCHS, MAE_LR, MAE_WEIGHT_DECAY,
    MAE_WARMUP_EPOCHS, MAE_MASK_RATIO, NUM_BANDS, TILE_SIZE,
)


def main():
    import torch
    from src.foundation.mae_model import MaskedAutoencoder
    from src.foundation.mae_pretrain import pretrain_mae
    from src.foundation.tile_dataset import SuratTileDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Dataset
    tiles_dir = RAW_DIR / "surat_tiles"
    if not tiles_dir.exists() or len(list(tiles_dir.glob("*.pt"))) == 0:
        logger.warning(f"No tiles found at {tiles_dir}. Run download_surat_tiles.py first.")
        logger.info("Using synthetic demo dataset...")
        tiles_dir.mkdir(parents=True, exist_ok=True)
        import torch as t
        for i in range(50):
            t.save(t.randn(NUM_BANDS, TILE_SIZE, TILE_SIZE).clamp(0, 1),
                   tiles_dir / f"tile_{i:04d}.pt")

    dataset = SuratTileDataset(str(tiles_dir))
    logger.info(f"Dataset: {len(dataset)} tiles")

    # Model
    model = MaskedAutoencoder(
        in_channels=NUM_BANDS,
        img_size=TILE_SIZE,
    )
    logger.info(f"MAE model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Pre-train
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "mae_pretrained.pt"

    pretrain_mae(
        model=model,
        dataset=dataset,
        epochs=MAE_EPOCHS,
        lr=MAE_LR,
        weight_decay=MAE_WEIGHT_DECAY,
        warmup_epochs=MAE_WARMUP_EPOCHS,
        mask_ratio=MAE_MASK_RATIO,
        device=device,
        checkpoint_path=str(checkpoint_path),
    )

    logger.info(f"Pre-training complete. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
