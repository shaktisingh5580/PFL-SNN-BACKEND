"""
Image preprocessing: normalization, compositing, alignment.
Prepares temporal pairs for the Siamese-SNN model.
"""
import logging
from typing import Tuple, Optional, List

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Per-band statistics for Sentinel-2 L2A (approximate, from global datasets)
# These are reflectance scale factors for normalization
S2_BAND_STATS = {
    "mean": np.array([
        1354.40, 1118.24, 1042.13, 947.62, 1199.47, 2003.07,
        2374.01, 2301.35, 2599.78, 732.08, 12.11, 1819.74, 1118.28
    ], dtype=np.float32),
    "std": np.array([
        245.71, 333.00, 395.09, 593.75, 566.40, 861.18,
        1086.63, 1117.98, 1172.16, 404.91, 4.77, 1002.58, 761.30
    ], dtype=np.float32),
    "min": np.array([0.0] * 13, dtype=np.float32),
    "max": np.array([10000.0] * 13, dtype=np.float32),
}


def normalize_bands(
    bands: np.ndarray,
    method: str = "minmax",
    clip_percentile: Optional[Tuple[float, float]] = (2, 98),
) -> np.ndarray:
    """
    Normalize multi-band image.

    Args:
        bands: (C, H, W) raw reflectance values
        method: 'minmax' | 'standardize' | 'percentile'
        clip_percentile: (low, high) percentile for clipping before normalization

    Returns:
        Normalized bands (C, H, W) in [0, 1] range
    """
    bands = bands.astype(np.float32).copy()

    if method == "minmax":
        for c in range(bands.shape[0]):
            band = bands[c]
            if clip_percentile:
                lo = np.percentile(band[band > 0], clip_percentile[0]) if (band > 0).any() else 0
                hi = np.percentile(band[band > 0], clip_percentile[1]) if (band > 0).any() else 1
                band = np.clip(band, lo, hi)
            bmin, bmax = band.min(), band.max()
            if bmax - bmin > 0:
                bands[c] = (band - bmin) / (bmax - bmin)
            else:
                bands[c] = np.zeros_like(band)

    elif method == "standardize":
        for c in range(bands.shape[0]):
            bands[c] = (bands[c] - S2_BAND_STATS["mean"][c]) / (S2_BAND_STATS["std"][c] + 1e-8)

    elif method == "percentile":
        for c in range(bands.shape[0]):
            band = bands[c]
            lo = np.percentile(band[band > 0], 2) if (band > 0).any() else 0
            hi = np.percentile(band[band > 0], 98) if (band > 0).any() else 1
            bands[c] = np.clip((band - lo) / (hi - lo + 1e-8), 0, 1)

    return bands


def create_median_composite(
    image_stack: List[np.ndarray],
    cloud_masks: List[np.ndarray],
) -> np.ndarray:
    """
    Create a median composite from multiple cloud-masked images.

    Args:
        image_stack: List of (C, H, W) arrays
        cloud_masks: List of (H, W) boolean masks (True = cloud)

    Returns:
        Median composite (C, H, W)
    """
    if not image_stack:
        raise ValueError("Empty image stack")

    C, H, W = image_stack[0].shape
    stack = np.stack(image_stack, axis=0)  # (N, C, H, W)
    masks = np.stack(cloud_masks, axis=0)  # (N, H, W)

    # Expand mask to match bands dimension
    masks_expanded = masks[:, np.newaxis, :, :]  # (N, 1, H, W)
    masks_expanded = np.broadcast_to(masks_expanded, stack.shape)

    # Create masked array
    masked_stack = np.ma.array(stack, mask=masks_expanded)

    # Compute median ignoring masked values
    composite = np.ma.median(masked_stack, axis=0).filled(0.0)

    logger.info(f"Created median composite from {len(image_stack)} images")
    return composite.astype(np.float32)


def align_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    transform_a=None,
    transform_b=None,
    crs_a: str = "EPSG:4326",
    crs_b: str = "EPSG:4326",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verify and align two temporal images to the same grid.

    Args:
        img_a: (C, H, W) first image
        img_b: (C, H, W) second image
        transform_a, transform_b: rasterio Affine transforms
        crs_a, crs_b: Coordinate reference system strings

    Returns:
        Tuple of aligned (C, H, W) arrays
    """
    # If same CRS and same shape, assume aligned
    if crs_a == crs_b and img_a.shape == img_b.shape:
        logger.info("Images already aligned (same CRS and shape)")
        return img_a, img_b

    # Different shapes — resize to match
    if img_a.shape != img_b.shape:
        logger.warning(
            f"Shape mismatch: {img_a.shape} vs {img_b.shape}. "
            "Resizing img_b to match img_a."
        )
        from scipy.ndimage import zoom

        C_a, H_a, W_a = img_a.shape
        C_b, H_b, W_b = img_b.shape

        zoom_factors = (1, H_a / H_b, W_a / W_b)
        img_b = zoom(img_b, zoom_factors, order=1).astype(np.float32)

    # If CRS differs and rasterio is available, reproject
    if crs_a != crs_b and HAS_RASTERIO and transform_a and transform_b:
        logger.info(f"Reprojecting from {crs_b} to {crs_a}")
        C, H, W = img_a.shape
        img_b_reproj = np.zeros_like(img_a)
        for c in range(C):
            reproject(
                source=img_b[c],
                destination=img_b_reproj[c],
                src_transform=transform_b,
                src_crs=crs_b,
                dst_transform=transform_a,
                dst_crs=crs_a,
                resampling=Resampling.bilinear,
            )
        img_b = img_b_reproj

    return img_a, img_b


def prepare_model_input(
    before_patch: dict,
    after_patch: dict,
    normalize_method: str = "minmax",
    target_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: cloud mask → normalize → align → prepare tensors.

    Args:
        before_patch: Dict from fetcher (keys: 'bands', 'scl', etc.)
        after_patch: Dict from fetcher
        normalize_method: Normalization method
        target_size: Target spatial dimension

    Returns:
        (img_before, img_after) as (C, H, W) float32 arrays ready for model
    """
    from .cloud_mask import create_combined_cloud_mask, apply_cloud_mask

    # Apply cloud masking
    mask_before = create_combined_cloud_mask(before_patch["bands"], before_patch["scl"])
    mask_after = create_combined_cloud_mask(after_patch["bands"], after_patch["scl"])

    img_before = apply_cloud_mask(before_patch["bands"], mask_before)
    img_after = apply_cloud_mask(after_patch["bands"], mask_after)

    # Normalize
    img_before = normalize_bands(img_before, method=normalize_method)
    img_after = normalize_bands(img_after, method=normalize_method)

    # Align
    img_before, img_after = align_images(img_before, img_after)

    # Resize if needed
    _, H, W = img_before.shape
    if H != target_size or W != target_size:
        from scipy.ndimage import zoom
        zoom_h = target_size / H
        zoom_w = target_size / W
        img_before = zoom(img_before, (1, zoom_h, zoom_w), order=1).astype(np.float32)
        img_after = zoom(img_after, (1, zoom_h, zoom_w), order=1).astype(np.float32)

    logger.info(f"Prepared model input: {img_before.shape}")
    return img_before, img_after
