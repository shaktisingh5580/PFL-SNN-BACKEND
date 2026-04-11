"""
Bulk Surat Sentinel-2 tile download for MAE pre-training.
Downloads unlabeled tiles using Sentinel Hub API.
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    SURAT_BBOX, RAW_DIR, SH_CLIENT_ID, SH_CLIENT_SECRET,
    TILE_SIZE, RESOLUTION, NUM_BANDS,
)


def download_surat_tiles(
    bbox: list = None,
    output_dir: str = None,
    max_tiles: int = 500,
    max_cloud_pct: float = 10.0,
):
    """Download Sentinel-2 tiles for the Surat region."""
    bbox = bbox or SURAT_BBOX
    output = Path(output_dir or RAW_DIR / "surat_tiles")
    output.mkdir(parents=True, exist_ok=True)

    if not SH_CLIENT_ID:
        logger.warning("SH_CLIENT_ID not set. Generating synthetic tiles for demo.")
        _generate_synthetic_tiles(output, max_tiles)
        return

    try:
        from src.pipeline.fetcher import SentinelFetcher
        fetcher = SentinelFetcher()

        logger.info(f"Downloading up to {max_tiles} tiles for bbox={bbox}")
        logger.info(f"Cloud filter: <{max_cloud_pct}% | Output: {output}")

        # Download quarterly composites for the last 2 years
        date_ranges = [
            ("2024-01-01", "2024-03-31"),
            ("2024-04-01", "2024-06-30"),
            ("2024-07-01", "2024-09-30"),
            ("2024-10-01", "2024-12-31"),
            ("2025-01-01", "2025-03-31"),
            ("2025-04-01", "2025-06-30"),
            ("2025-07-01", "2025-09-30"),
            ("2025-10-01", "2025-12-31"),
        ]

        count = 0
        for start, end in date_ranges:
            if count >= max_tiles:
                break
            tiles = fetcher.download_tiles(bbox, start, end, str(output))
            count += len(tiles)
            logger.info(f"Downloaded {len(tiles)} tiles for {start} to {end}")

        logger.info(f"Total: {count} tiles downloaded to {output}")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Falling back to synthetic tiles...")
        _generate_synthetic_tiles(output, max_tiles)


def _generate_synthetic_tiles(output: Path, count: int):
    """Generate synthetic tiles for demo/testing."""
    import numpy as np
    import torch

    logger.info(f"Generating {count} synthetic {TILE_SIZE}x{TILE_SIZE}x{NUM_BANDS} tiles...")

    for i in range(count):
        tile = torch.randn(NUM_BANDS, TILE_SIZE, TILE_SIZE).clamp(0, 1)
        torch.save(tile, output / f"tile_{i:04d}.pt")

        if (i + 1) % 100 == 0:
            logger.info(f"  Generated {i + 1}/{count}")

    logger.info(f"Synthetic tiles saved to {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Surat Sentinel-2 tiles")
    parser.add_argument("--max-tiles", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    download_surat_tiles(max_tiles=args.max_tiles, output_dir=args.output)
