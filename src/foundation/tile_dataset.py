"""
Unlabeled Sentinel-2 tile dataset for MAE pre-training.
"""
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class SentinelTileDataset(Dataset):
    """Dataset of unlabeled Sentinel-2 tiles for self-supervised pre-training."""

    def __init__(
        self,
        root_dir: str,
        img_size: int = 512,
        num_bands: int = 13,
        augment: bool = True,
        max_cloud_pct: float = 10.0,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.num_bands = num_bands
        self.augment = augment

        # Find all GeoTIFF tiles
        self.tile_paths = []
        if self.root_dir.exists():
            for ext in ["*.tif", "*.tiff"]:
                self.tile_paths.extend(sorted(self.root_dir.glob(f"**/{ext}")))

        if not self.tile_paths:
            logger.warning(f"No tiles found in {root_dir}. Using synthetic data.")
            self.use_synthetic = True
            self.tile_paths = list(range(500))  # 500 synthetic tiles
        else:
            self.use_synthetic = False

        logger.info(f"SentinelTileDataset: {len(self.tile_paths)} tiles")

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        if self.use_synthetic:
            return self._get_synthetic_tile(idx)

        path = self.tile_paths[idx]
        try:
            if HAS_RASTERIO:
                with rasterio.open(str(path)) as src:
                    bands = src.read()  # (C, H, W)
            else:
                bands = np.random.rand(self.num_bands, self.img_size, self.img_size).astype(np.float32)
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return self._get_synthetic_tile(idx)

        # Ensure correct number of bands
        C, H, W = bands.shape
        if C < self.num_bands:
            pad = np.zeros((self.num_bands - C, H, W), dtype=bands.dtype)
            bands = np.concatenate([bands, pad], axis=0)
        elif C > self.num_bands:
            bands = bands[:self.num_bands]

        # Random crop to img_size
        if H > self.img_size and W > self.img_size:
            y = np.random.randint(0, H - self.img_size)
            x = np.random.randint(0, W - self.img_size)
            bands = bands[:, y:y + self.img_size, x:x + self.img_size]
        elif H != self.img_size or W != self.img_size:
            # Resize
            from scipy.ndimage import zoom
            zoom_h = self.img_size / H
            zoom_w = self.img_size / W
            bands = zoom(bands, (1, zoom_h, zoom_w), order=1)

        # Normalize to [0, 1]
        bands = bands.astype(np.float32)
        for c in range(bands.shape[0]):
            bmin, bmax = bands[c].min(), bands[c].max()
            if bmax > bmin:
                bands[c] = (bands[c] - bmin) / (bmax - bmin)

        if self.augment:
            bands = self._augment(bands)

        return torch.from_numpy(bands.copy()).float()

    def _augment(self, bands: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.5:
            bands = np.flip(bands, axis=2)
        if np.random.random() > 0.5:
            bands = np.flip(bands, axis=1)
        k = np.random.randint(0, 4)
        if k > 0:
            bands = np.rot90(bands, k, axes=(1, 2))
        # Spectral jitter
        if np.random.random() > 0.5:
            jitter = 1.0 + (np.random.rand(bands.shape[0], 1, 1) - 0.5) * 0.1
            bands = bands * jitter
            bands = np.clip(bands, 0, 1)
        return bands

    def _get_synthetic_tile(self, idx: int) -> torch.Tensor:
        """Generate a synthetic tile with realistic spectral patterns."""
        np.random.seed(idx)
        C, H, W = self.num_bands, self.img_size, self.img_size
        bands = np.random.rand(C, H, W).astype(np.float32) * 0.4 + 0.1

        # Add spatial structure
        yy, xx = np.mgrid[:H, :W]
        for _ in range(np.random.randint(3, 10)):
            cy, cx = np.random.randint(0, H), np.random.randint(0, W)
            radius = np.random.randint(20, 100)
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius ** 2
            land_type = np.random.choice(["vegetation", "urban", "water"])
            if land_type == "vegetation":
                bands[3, mask] *= 0.4  # Low red
                bands[7, mask] *= 1.8  # High NIR
            elif land_type == "urban":
                bands[11, mask] *= 1.5  # High SWIR
                bands[7, mask] *= 0.7
            else:
                bands[1, mask] *= 0.3  # Low blue
                bands[2, mask] *= 1.5  # High green

        bands = np.clip(bands, 0, 1)
        return torch.from_numpy(bands).float()
