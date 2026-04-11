"""
Clean, CPU-optimized training entry point for the Siamese-SNN change detection model.

Usage:
    python scripts/train_backend.py [--epochs N] [--batch-size N] [--resume PATH]

Automatically detects OSCD dataset in data/oscd/. Falls back to synthetic data
for a quick smoke-test if the dataset is not yet downloaded.
"""
import sys
import logging
import argparse
import time
from pathlib import Path

# Make sure imports resolve from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CPU-tuned defaults (override via CLI) ──────────────────────────────────
CPU_DEFAULTS = {
    "epochs": 30,
    "batch_size": 2,
    "lr": 3e-4,              # Lower LR for stable convergence
    "weight_decay": 1e-4,
    "patch_size": 128,       # 128x128 patches fit comfortably in RAM
    "num_steps": 8,          # SNN time-steps: 8 for better spike dynamics
    "num_bands": 4,          # Use only B02,B03,B04,B08 (RGB+NIR) – 4 key bands
    "encoder_channels": [32, 64, 128, 256],  # Smaller encoder for CPU
    "gradient_clip": 1.0,
    "change_weight": 15.0,   # Aggressive weight for minority class (~3-5% pixels)
    "predict_threshold": 0.3,  # Lower threshold to catch faint change signals
}
BANDS_4 = ["B02", "B03", "B04", "B08"]   # RGB + NIR


def parse_args():
    p = argparse.ArgumentParser(description="Train Siamese-SNN on OSCD dataset (CPU mode)")
    p.add_argument("--epochs",     type=int,   default=CPU_DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int,   default=CPU_DEFAULTS["batch_size"])
    p.add_argument("--lr",         type=float, default=CPU_DEFAULTS["lr"])
    p.add_argument("--resume",     type=str,   default=None, help="Path to checkpoint .pt to resume from")
    p.add_argument("--smoke-test", action="store_true", help="2 epochs of synthetic data to verify the pipeline")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    from src.model.siamese_snn import SiameseSNN
    from src.model.losses import CombinedLoss
    from src.training.dataset import OSCDDataset

    MODELS_DIR = ROOT / "outputs" / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OSCD_DIR = ROOT / "data" / "oscd"

    device = torch.device("cpu")
    logger.info("=" * 60)
    logger.info("  Siamese-SNN Training  |  CPU Mode")
    logger.info("=" * 60)

    # ── Dataset ────────────────────────────────────────────────────────────
    if args.smoke_test:
        logger.info("SMOKE TEST MODE — using synthetic data (2 epochs)")
        n_train, n_val = 10, 4
        P, C = CPU_DEFAULTS["patch_size"], CPU_DEFAULTS["num_bands"]
        train_ds = TensorDataset(
            torch.randn(n_train, C, P, P),
            torch.randn(n_train, C, P, P),
            torch.randint(0, 2, (n_train, P, P)),
        )
        val_ds = TensorDataset(
            torch.randn(n_val, C, P, P),
            torch.randn(n_val, C, P, P),
            torch.randint(0, 2, (n_val, P, P)),
        )
        args.epochs = 2

    else:
        oscd_has_data = any((OSCD_DIR / city).exists() for city in [
            "aguasclaras", "bercy", "bordeaux", "nantes", "paris", "rennes",
            "saclay_e", "abudhabi", "cupertino", "pisa", "beihai",
            "hongkong", "beirut", "mumbai",
        ])

        if not oscd_has_data:
            logger.warning(
                "\n" + "=" * 60 +
                "\n  OSCD dataset not found in data/oscd/" +
                "\n  Run: python scripts/download_oscd.py" +
                "\n  Or place the OSCD folders directly in data/oscd/" +
                "\n  Falling back to SYNTHETIC data for a smoke-test." +
                "\n" + "=" * 60
            )
            n_train, n_val = 10, 4
            P, C = CPU_DEFAULTS["patch_size"], CPU_DEFAULTS["num_bands"]
            train_ds = TensorDataset(
                torch.randn(n_train, C, P, P),
                torch.randn(n_train, C, P, P),
                torch.randint(0, 2, (n_train, P, P)),
            )
            val_ds = TensorDataset(
                torch.randn(n_val, C, P, P),
                torch.randn(n_val, C, P, P),
                torch.randint(0, 2, (n_val, P, P)),
            )
        else:
            logger.info(f"Loading OSCD dataset from {OSCD_DIR}")
            train_ds = OSCDDataset(
                str(OSCD_DIR),
                split="train",
                patch_size=CPU_DEFAULTS["patch_size"],
                bands=BANDS_4,
                augment=True,
            )
            val_ds = OSCDDataset(
                str(OSCD_DIR),
                split="val",
                patch_size=CPU_DEFAULTS["patch_size"],
                bands=BANDS_4,
                augment=False,
            )
            logger.info(f"OSCD loaded: {len(train_ds)} train patches, {len(val_ds)} val patches")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,   # 0 for Windows CPU compatibility
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = SiameseSNN(
        in_channels=CPU_DEFAULTS["num_bands"],
        encoder_channels=CPU_DEFAULTS["encoder_channels"],
        beta=0.9,
        num_steps=CPU_DEFAULTS["num_steps"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters  (CPU mode — reduced size)")

    # ── Loss / Optimizer ───────────────────────────────────────────────────
    loss_fn = CombinedLoss(alpha=0.5, gamma=0.2, change_weight=CPU_DEFAULTS["change_weight"])
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=CPU_DEFAULTS["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 1
    best_f1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        try:
            scheduler.load_state_dict(ckpt.get("scheduler_state_dict", {}))
        except Exception:
            logger.warning("Scheduler state incompatible (changed type), using fresh scheduler")
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']} (best F1: {best_f1:.4f})")

    # ── Training Loop ──────────────────────────────────────────────────────
    history = {"train_loss": [], "val_f1": [], "val_iou": []}
    T = CPU_DEFAULTS["num_steps"]

    logger.info(f"Starting training: {args.epochs} epochs, batch={args.batch_size}, LR={args.lr}")
    logger.info(f"SNN time-steps={T}, patch={CPU_DEFAULTS['patch_size']}x{CPU_DEFAULTS['patch_size']}, bands={CPU_DEFAULTS['num_bands']}")
    logger.info("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # — Train —
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 3:
                img_a, img_b, mask = batch
            else:
                img_a, img_b, mask = batch[0], batch[1], batch[2]

            img_a = img_a.to(device).float()
            img_b = img_b.to(device).float()
            mask  = mask.to(device).long()

            optimizer.zero_grad()
            spk_rec, _ = model(img_a, img_b)
            loss = loss_fn(spk_rec, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CPU_DEFAULTS["gradient_clip"])
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_train_loss)

        # — Validate —
        model.eval()
        tp = fp = fn = tn = 0
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img_a, img_b, mask = batch[0].float(), batch[1].float(), batch[2].long()
                spk_rec, _ = model(img_a, img_b)
                val_loss_total += loss_fn(spk_rec, mask).item()
                pred = model.predict(img_a, img_b, threshold=CPU_DEFAULTS["predict_threshold"])
                tp += ((pred == 1) & (mask == 1)).sum().item()
                fp += ((pred == 1) & (mask == 0)).sum().item()
                fn += ((pred == 0) & (mask == 1)).sum().item()
                tn += ((pred == 0) & (mask == 0)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        iou       = tp / (tp + fp + fn + 1e-8)
        avg_val_loss = val_loss_total / max(len(val_loader), 1)
        history["val_f1"].append(f1)
        history["val_iou"].append(iou)

        scheduler.step(f1)
        elapsed = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"F1: {f1:.4f} | IoU: {iou:.4f} | "
            f"Time: {elapsed:.0f}s"
        )

        # — Save best —
        if f1 > best_f1:
            best_f1 = f1
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": {"f1": f1, "iou": iou},
                "best_f1": best_f1,
                "config": CPU_DEFAULTS,
            }
            torch.save(checkpoint, str(MODELS_DIR / "best_model.pt"))
            logger.info(f"  ★ New best model saved! F1={best_f1:.4f}")

        # — Periodic checkpoint every 5 epochs —
        if epoch % 5 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": {"f1": f1, "iou": iou},
                "best_f1": best_f1,
                "config": CPU_DEFAULTS,
            }
            torch.save(checkpoint, str(MODELS_DIR / f"checkpoint_epoch_{epoch}.pt"))
            logger.info(f"  → Checkpoint saved: checkpoint_epoch_{epoch}.pt")

    logger.info("=" * 60)
    logger.info(f"Training complete!  Best F1: {best_f1:.4f}")
    logger.info(f"Best model → {MODELS_DIR / 'best_model.pt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
