"""
Cloud masking using s2cloudless and Sentinel-2 SCL band.
Produces clean binary masks for cloud-free pixel selection.
"""
import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from s2cloudless import S2PixelCloudDetector
    HAS_S2CLOUDLESS = True
except ImportError:
    HAS_S2CLOUDLESS = False
    logger.warning("s2cloudless not installed. Using SCL-only cloud masking.")


def create_cloud_mask_s2cloudless(
    bands: np.ndarray,
    threshold: float = 0.4,
    average_over: int = 4,
    dilation_size: int = 2,
) -> np.ndarray:
    """
    Generate cloud mask using s2cloudless.

    Args:
        bands: (C, H, W) Sentinel-2 bands (need B01,B02,B04,B05,B08,B8A,B09,B10,B11,B12)
        threshold: Cloud probability threshold
        average_over: Averaging window size
        dilation_size: Dilation size for cloud mask

    Returns:
        Binary cloud mask (H, W), True = cloud
    """
    if not HAS_S2CLOUDLESS:
        logger.warning("s2cloudless unavailable, falling back to SCL-based masking")
        return np.zeros(bands.shape[1:], dtype=bool)

    cloud_detector = S2PixelCloudDetector(
        threshold=threshold,
        average_over=average_over,
        dilation_size=dilation_size,
        all_bands=True,
    )

    # s2cloudless expects (1, H, W, C) — batch of 1
    bands_hwc = np.transpose(bands, (1, 2, 0))[np.newaxis, ...]

    cloud_prob = cloud_detector.get_cloud_probability_maps(bands_hwc)
    cloud_mask = cloud_detector.get_cloud_masks(bands_hwc)

    return cloud_mask.squeeze().astype(bool)


def create_cloud_mask_scl(
    scl: np.ndarray,
    cloud_classes: list = None,
    shadow_classes: list = None,
) -> np.ndarray:
    """
    Generate cloud mask from Scene Classification Layer (SCL).

    SCL Classes:
        0: No_Data, 1: Saturated/Defective, 2: Dark_Area_Pixels,
        3: Cloud_Shadows, 4: Vegetation, 5: Not_Vegetated,
        6: Water, 7: Unclassified, 8: Cloud_Medium_Probability,
        9: Cloud_High_Probability, 10: Thin_Cirrus, 11: Snow

    Args:
        scl: (H, W) Scene Classification Layer
        cloud_classes: SCL class IDs considered as cloud
        shadow_classes: SCL class IDs considered as shadow

    Returns:
        Binary mask (H, W), True = cloud/shadow
    """
    if cloud_classes is None:
        cloud_classes = [8, 9, 10]  # Cloud medium, high, cirrus
    if shadow_classes is None:
        shadow_classes = [3]  # Cloud shadows

    bad_classes = cloud_classes + shadow_classes + [0, 1]  # Also mask no_data & defective

    mask = np.isin(scl, bad_classes)
    return mask


def create_combined_cloud_mask(
    bands: np.ndarray,
    scl: np.ndarray,
    threshold: float = 0.4,
) -> np.ndarray:
    """
    Combine s2cloudless and SCL masks for robust cloud masking.

    Args:
        bands: (C, H, W) all 13 Sentinel-2 bands
        scl: (H, W) Scene Classification Layer
        threshold: Cloud probability threshold

    Returns:
        Binary mask (H, W), True = cloud/shadow/bad pixel
    """
    scl_mask = create_cloud_mask_scl(scl)

    if HAS_S2CLOUDLESS:
        s2c_mask = create_cloud_mask_s2cloudless(bands, threshold=threshold)
        # Union — mask if either detector flags it
        combined = scl_mask | s2c_mask
    else:
        combined = scl_mask

    cloud_pct = combined.sum() / combined.size * 100
    logger.info(f"Cloud/shadow mask: {cloud_pct:.1f}% of pixels masked")

    return combined


def get_cloud_percentage(mask: np.ndarray) -> float:
    """Return the percentage of cloudy pixels."""
    return float(mask.sum() / mask.size * 100)


def apply_cloud_mask(bands: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Apply cloud mask to multi-band image.

    Args:
        bands: (C, H, W) image bands
        mask: (H, W) binary cloud mask (True = cloud)
        fill_value: Value to fill masked pixels

    Returns:
        Masked bands (C, H, W) with clouds set to fill_value
    """
    masked = bands.copy()
    masked[:, mask] = fill_value
    return masked
