"""
Change Type Classifier for Satellite Imagery Change Detection.

Uses spectral indices (NDVI, NDBI, MNDWI) to categorize detected changes
into: Construction, Vegetation Loss, Water Change, or Other.

Works with either 4 bands (B02, B03, B04, B08) or 5 bands (+B11 SWIR).
When B11 is unavailable, uses a simplified 4-band classification.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional

try:
    from scipy.ndimage import binary_opening, binary_closing, binary_dilation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ChangeTypeClassifier:
    """
    Classifies binary change masks into specific change categories using
    Sentinel-2 spectral indices.
    
    Supports two modes:
    - 5-band mode (B02, B03, B04, B08, B11): Full NDVI + NDBI + MNDWI
    - 4-band mode (B02, B03, B04, B08): NDVI-only classification (automatic fallback)
    """

    CLASSES = {
        0: "No Change",
        1: "Construction / Urban Sprawl",
        2: "Vegetation Clearance / Deforestation",
        3: "Water Body Change / Sand Mining",
        4: "Other Land Alteration"
    }

    # Colors for visualization (RGB)
    COLORS = {
        0: [0, 0, 0],           # Black
        1: [230, 50, 50],       # Red — Construction
        2: [50, 200, 50],       # Green — Vegetation Loss
        3: [50, 130, 255],      # Blue — Water Change
        4: [240, 200, 50]       # Yellow — Other
    }

    # Compliance violation severity
    SEVERITY = {
        0: "none",
        1: "high",     # Construction often requires permits
        2: "medium",   # Vegetation loss may violate green zone rules
        3: "high",     # Water body alteration is typically illegal
        4: "low"       # Minor alteration, needs review
    }

    def __init__(
        self,
        ndvi_drop_threshold: float = -0.08,
        ndbi_rise_threshold: float = 0.03,
        mndwi_threshold: float = 0.06,
        min_cluster_pixels: int = 5,
        morphological_cleanup: bool = True,
    ):
        """
        Args:
            ndvi_drop_threshold: NDVI decrease that signals vegetation loss (negative value)
            ndbi_rise_threshold: NDBI increase that signals construction (positive value)
            mndwi_threshold: |MNDWI| change that signals water alteration
            min_cluster_pixels: Minimum cluster size to keep (removes noise)
            morphological_cleanup: Apply opening/closing to clean mask
        """
        self.ndvi_thresh = ndvi_drop_threshold
        self.ndbi_thresh = ndbi_rise_threshold
        self.mndwi_thresh = mndwi_threshold
        self.min_cluster = min_cluster_pixels
        self.do_cleanup = morphological_cleanup

    # ─── Spectral Index Calculations ─────────────────────────────

    @staticmethod
    def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Normalized difference: (a - b) / (a + b), safe from div-by-zero."""
        denom = a + b
        # Where both bands are zero, return 0
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(np.abs(denom) > 1e-10, (a - b) / denom, 0.0)
        return np.clip(result, -1.0, 1.0)

    def compute_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """NDVI = (NIR - Red) / (NIR + Red). High = vegetation."""
        return self._safe_ratio(nir, red)

    def compute_ndbi(self, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """NDBI = (SWIR - NIR) / (SWIR + NIR). High = built-up."""
        return self._safe_ratio(swir, nir)

    def compute_mndwi(self, green: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """MNDWI = (Green - SWIR) / (Green + SWIR). High = water."""
        return self._safe_ratio(green, swir)

    def compute_brightness_change(self, img_before: np.ndarray, img_after: np.ndarray) -> np.ndarray:
        """
        Simple brightness change across all bands.
        Useful as a 4-band fallback for construction detection.
        """
        mean_before = img_before.mean(axis=0)
        mean_after = img_after.mean(axis=0)
        return mean_after - mean_before

    # ─── Mask Cleanup ────────────────────────────────────────────

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Remove noise from binary mask using morphological operations."""
        if not HAS_SCIPY or not self.do_cleanup:
            return mask

        # Close small gaps, then open to remove tiny noise
        struct = np.ones((3, 3))
        cleaned = binary_closing(mask, structure=struct, iterations=1)
        cleaned = binary_opening(cleaned, structure=struct, iterations=1)
        return cleaned.astype(mask.dtype)

    # ─── Main Classification ─────────────────────────────────────

    def classify(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray,
        change_mask: np.ndarray,
        band_names: list = None,
    ) -> np.ndarray:
        """
        Classify change pixels into categories.

        Args:
            img_before: (C, H, W) array, time T1
            img_after:  (C, H, W) array, time T2
            change_mask: (H, W) binary mask from SNN model
            band_names: list of band names matching channels, e.g. ['B02','B03','B04','B08']

        Returns:
            class_map: (H, W) uint8 array with class IDs
        """
        # Convert tensors
        if torch.is_tensor(img_before): img_before = img_before.cpu().numpy()
        if torch.is_tensor(img_after): img_after = img_after.cpu().numpy()
        if torch.is_tensor(change_mask): change_mask = change_mask.cpu().numpy()

        if band_names is None:
            band_names = ['B02', 'B03', 'B04', 'B08']

        # Build band index lookup
        idx = {b: i for i, b in enumerate(band_names)}
        has_swir = 'B11' in idx

        # Clean the change mask first
        clean_change = self.clean_mask(change_mask)
        changed = (clean_change > 0)

        h, w = change_mask.shape
        class_map = np.zeros((h, w), dtype=np.uint8)

        if not changed.any():
            return class_map

        # ── NDVI (always available: needs B04=Red, B08=NIR) ──
        red_b = img_before[idx['B04']]
        nir_b = img_before[idx['B08']]
        red_a = img_after[idx['B04']]
        nir_a = img_after[idx['B08']]

        ndvi_before = self.compute_ndvi(red_b, nir_b)
        ndvi_after = self.compute_ndvi(red_a, nir_a)
        d_ndvi = ndvi_after - ndvi_before

        if has_swir:
            # ── Full 5-band classification ──
            green_b = img_before[idx['B03']]
            swir_b = img_before[idx['B11']]
            green_a = img_after[idx['B03']]
            swir_a = img_after[idx['B11']]

            ndbi_before = self.compute_ndbi(nir_b, swir_b)
            ndbi_after = self.compute_ndbi(nir_a, swir_a)
            d_ndbi = ndbi_after - ndbi_before

            mndwi_before = self.compute_mndwi(green_b, swir_b)
            mndwi_after = self.compute_mndwi(green_a, swir_a)
            d_mndwi = mndwi_after - mndwi_before

            # Rule 1: Construction — veg drops AND built-up rises
            is_construction = changed & (d_ndvi <= self.ndvi_thresh) & (d_ndbi >= self.ndbi_thresh)

            # Rule 2: Vegetation Loss — veg drops but NOT construction
            is_veg_loss = changed & (~is_construction) & (d_ndvi <= self.ndvi_thresh)

            # Rule 3: Water change — MNDWI shifts significantly
            is_water = changed & (~is_construction) & (~is_veg_loss) & (np.abs(d_mndwi) >= self.mndwi_thresh)

        else:
            # ── 4-band fallback (no SWIR) ──
            # Use brightness increase as proxy for construction
            d_brightness = self.compute_brightness_change(img_before, img_after)

            # Rule 1: Construction — veg drops AND area gets brighter (concrete is bright)
            is_construction = changed & (d_ndvi <= self.ndvi_thresh) & (d_brightness > 0.05)

            # Rule 2: Vegetation Loss
            is_veg_loss = changed & (~is_construction) & (d_ndvi <= self.ndvi_thresh)

            # Rule 3: Water proxy — using green band ratio (B03)
            green_ratio = self._safe_ratio(
                img_after[idx['B03']], img_before[idx['B03']]
            )
            is_water = changed & (~is_construction) & (~is_veg_loss) & (np.abs(green_ratio) > 0.15)

        # Rule 4: Other — anything left
        is_other = changed & (~is_construction) & (~is_veg_loss) & (~is_water)

        class_map[is_construction] = 1
        class_map[is_veg_loss] = 2
        class_map[is_water] = 3
        class_map[is_other] = 4

        return class_map

    # ─── Visualization ───────────────────────────────────────────

    def get_color_map(self, class_map: np.ndarray) -> np.ndarray:
        """Convert class IDs to RGB image for display."""
        h, w = class_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, color in self.COLORS.items():
            rgb[class_map == cid] = color
        return rgb

    def get_overlay(
        self,
        base_rgb: np.ndarray,
        class_map: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay the classification colors on top of a base RGB image.
        Only overlays where class_map > 0.
        """
        color_map = self.get_color_map(class_map).astype(np.float32) / 255.0

        if base_rgb.dtype == np.uint8:
            base = base_rgb.astype(np.float32) / 255.0
        else:
            base = base_rgb.copy()

        mask = class_map > 0
        result = base.copy()
        result[mask] = (1 - alpha) * base[mask] + alpha * color_map[mask]
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    # ─── Report ──────────────────────────────────────────────────

    def generate_report(self, class_map: np.ndarray, pixel_size_m: float = 10.0) -> dict:
        """
        Generate a compliance report from the classification.

        Args:
            class_map: (H, W) classification result
            pixel_size_m: Ground sampling distance in meters (Sentinel-2 = 10m)

        Returns:
            Dictionary with change statistics and compliance flags
        """
        pixel_area_m2 = pixel_size_m ** 2
        total_changed = (class_map > 0).sum()

        findings = []
        for cid, name in self.CLASSES.items():
            if cid == 0:
                continue
            count = int(np.sum(class_map == cid))
            if count > 0:
                area_m2 = count * pixel_area_m2
                area_ha = area_m2 / 10000.0
                findings.append({
                    "class_id": cid,
                    "class_name": name,
                    "pixel_count": count,
                    "area_m2": round(area_m2, 1),
                    "area_hectares": round(area_ha, 4),
                    "severity": self.SEVERITY[cid],
                    "percentage": round(count / max(total_changed, 1) * 100, 1),
                })

        return {
            "total_changed_pixels": int(total_changed),
            "total_changed_area_m2": round(total_changed * pixel_area_m2, 1),
            "findings": findings,
        }
