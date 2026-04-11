"""
Spectral index computation for Sentinel-2 imagery.
Computes NDVI, NDBI, MNDWI and their temporal differentials.
"""
import logging
from typing import Tuple, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel-2 band indices (0-indexed within 13-band stack)
BAND_MAP = {
    "B01": 0,   # Coastal aerosol (60m)
    "B02": 1,   # Blue (10m)
    "B03": 2,   # Green (10m)
    "B04": 3,   # Red (10m)
    "B05": 4,   # Vegetation Red Edge 1 (20m)
    "B06": 5,   # Vegetation Red Edge 2 (20m)
    "B07": 6,   # Vegetation Red Edge 3 (20m)
    "B08": 7,   # NIR (10m)
    "B8A": 8,   # Vegetation Red Edge 4 (20m)
    "B09": 9,   # Water Vapour (60m)
    "B10": 10,  # SWIR – Cirrus (60m)
    "B11": 11,  # SWIR 1 (20m)
    "B12": 12,  # SWIR 2 (20m)
}


def _safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safe division avoiding divide-by-zero."""
    return (a - b) / (a + b + eps)


def compute_ndvi(bands: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index.
    NDVI = (B8 - B4) / (B8 + B4)

    Args:
        bands: (C, H, W) or (13, H, W) Sentinel-2 bands

    Returns:
        NDVI array (H, W) in range [-1, 1]
    """
    nir = bands[BAND_MAP["B08"]]   # NIR
    red = bands[BAND_MAP["B04"]]   # Red
    ndvi = _safe_divide(nir, red)
    return np.clip(ndvi, -1, 1).astype(np.float32)


def compute_ndbi(bands: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Built-up Index.
    NDBI = (B11 - B8) / (B11 + B8)

    Args:
        bands: (C, H, W) Sentinel-2 bands

    Returns:
        NDBI array (H, W) in range [-1, 1]
    """
    swir = bands[BAND_MAP["B11"]]  # SWIR 1
    nir = bands[BAND_MAP["B08"]]   # NIR
    ndbi = _safe_divide(swir, nir)
    return np.clip(ndbi, -1, 1).astype(np.float32)


def compute_mndwi(bands: np.ndarray) -> np.ndarray:
    """
    Compute Modified Normalized Difference Water Index.
    MNDWI = (B3 - B11) / (B3 + B11)

    Args:
        bands: (C, H, W) Sentinel-2 bands

    Returns:
        MNDWI array (H, W) in range [-1, 1]
    """
    green = bands[BAND_MAP["B03"]]  # Green
    swir = bands[BAND_MAP["B11"]]   # SWIR 1
    mndwi = _safe_divide(green, swir)
    return np.clip(mndwi, -1, 1).astype(np.float32)


def compute_all_indices(bands: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute all spectral indices for a single image.

    Args:
        bands: (C, H, W) Sentinel-2 bands

    Returns:
        Dict with 'ndvi', 'ndbi', 'mndwi' arrays
    """
    return {
        "ndvi": compute_ndvi(bands),
        "ndbi": compute_ndbi(bands),
        "mndwi": compute_mndwi(bands),
    }


def compute_differential_indices(
    bands_before: np.ndarray,
    bands_after: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute temporal difference in spectral indices (ΔNDVI, ΔNDBI, ΔMNDWI).

    Args:
        bands_before: (C, H, W) before-image bands
        bands_after: (C, H, W) after-image bands

    Returns:
        Dict with 'delta_ndvi', 'delta_ndbi', 'delta_mndwi',
        plus individual 'before_*' and 'after_*' indices
    """
    idx_before = compute_all_indices(bands_before)
    idx_after = compute_all_indices(bands_after)

    deltas = {}
    for key in ["ndvi", "ndbi", "mndwi"]:
        deltas[f"before_{key}"] = idx_before[key]
        deltas[f"after_{key}"] = idx_after[key]
        deltas[f"delta_{key}"] = idx_after[key] - idx_before[key]

    return deltas


def generate_spectral_change_mask(
    deltas: Dict[str, np.ndarray],
    ndvi_threshold: float = -0.2,
    ndbi_threshold: float = 0.15,
    mndwi_threshold: float = -0.2,
) -> Dict[str, np.ndarray]:
    """
    Generate thresholded change maps from spectral differences.

    Args:
        deltas: Output of compute_differential_indices()
        ndvi_threshold: ΔNDVI threshold for vegetation loss (negative = loss)
        ndbi_threshold: ΔNDBI threshold for built-up increase (positive = increase)
        mndwi_threshold: ΔMNDWI threshold for water loss

    Returns:
        Dict with binary change masks for each type
    """
    masks = {
        "vegetation_loss": deltas["delta_ndvi"] < ndvi_threshold,
        "new_construction": (
            (deltas["delta_ndbi"] > ndbi_threshold) &
            (deltas["delta_ndvi"] < ndvi_threshold)
        ),
        "water_loss": deltas["delta_mndwi"] < mndwi_threshold,
        "built_up_increase": deltas["delta_ndbi"] > ndbi_threshold,
    }

    for name, mask in masks.items():
        pct = mask.sum() / mask.size * 100
        logger.info(f"Spectral change '{name}': {pct:.2f}% of pixels")

    return masks


def correlate_with_predictions(
    spectral_mask: np.ndarray,
    dl_predictions: np.ndarray,
) -> Dict[str, float]:
    """
    Compute correlation between spectral change maps and DL predictions.

    Args:
        spectral_mask: (H, W) binary spectral change mask
        dl_predictions: (H, W) binary DL prediction mask

    Returns:
        Dict with correlation metrics
    """
    spectral_flat = spectral_mask.flatten().astype(float)
    dl_flat = dl_predictions.flatten().astype(float)

    # Agreement metrics
    intersection = (spectral_mask & dl_predictions).sum()
    union = (spectral_mask | dl_predictions).sum()
    iou = intersection / (union + 1e-8)

    # Cohen's kappa
    agreement = (spectral_flat == dl_flat).mean()
    p_yes = spectral_flat.mean() * dl_flat.mean()
    p_no = (1 - spectral_flat.mean()) * (1 - dl_flat.mean())
    p_expected = p_yes + p_no
    kappa = (agreement - p_expected) / (1 - p_expected + 1e-8)

    # Pearson correlation
    if spectral_flat.std() > 0 and dl_flat.std() > 0:
        pearson = np.corrcoef(spectral_flat, dl_flat)[0, 1]
    else:
        pearson = 0.0

    return {
        "iou": float(iou),
        "agreement": float(agreement),
        "kappa": float(kappa),
        "pearson": float(pearson),
        "spectral_change_pct": float(spectral_flat.mean() * 100),
        "dl_change_pct": float(dl_flat.mean() * 100),
    }
