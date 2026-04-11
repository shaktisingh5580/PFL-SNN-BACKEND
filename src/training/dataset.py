"""
OSCD (Onera Satellite Change Detection) Dataset loader.
Loads bi-temporal Sentinel-2 image pairs with binary change masks.

Handles the real OSCD dataset structure where band files use long
Sentinel-2 naming conventions (e.g. S2A_OPER_MSI_..._B02.tif) and
bands may have different spatial resolutions (10m/20m/60m).
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from scipy.ndimage import zoom as scipy_zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Official OSCD labeled cities (have cm.png ground truth)
# 11 for training, 3 held out for validation (diverse geographies)
OSCD_TRAIN_CITIES = [
    "aguasclaras", "bercy", "bordeaux", "nantes", "paris", "rennes",
    "saclay_e", "cupertino", "pisa", "beihai", "hongkong"
]
OSCD_VAL_CITIES = [
    "abudhabi", "beirut", "mumbai"  # diverse validation cities WITH labels
]
# Unlabeled test cities (for inference only — NO cm.png labels)
OSCD_TEST_CITIES = [
    "brasilia", "montpellier", "norcia", "rio", "saclay_w",
    "valencia", "dubai", "lasvegas", "milano", "chongqing"
]

# Sentinel-2 band names in OSCD dataset ordering
OSCD_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


def _find_band_file(directory: Path, band_name: str) -> Optional[Path]:
    """
    Find a band TIF file in a directory, handling both simple naming (B02.tif)
    and long Sentinel-2 naming (S2A_OPER_MSI_..._B02.tif).
    """
    # Try simple name first
    simple = directory / f"{band_name}.tif"
    if simple.exists():
        return simple

    # Try glob for long Sentinel-2 naming convention
    matches = list(directory.glob(f"*_{band_name}.tif"))
    if matches:
        return matches[0]

    # Try case-insensitive glob
    matches = list(directory.glob(f"*_{band_name.lower()}.tif"))
    if not matches:
        matches = list(directory.glob(f"*_{band_name.upper()}.tif"))
    if matches:
        return matches[0]

    return None


class OSCDDataset(Dataset):
    """
    PyTorch Dataset for the Onera Satellite Change Detection dataset.

    Handles the real OSCD directory structure:
    oscd/
    ├── {city_name}/
    │   ├── imgs_1/         # Before images (per-band TIFFs)
    │   │   ├── S2A_..._B02.tif   (or B02.tif)
    │   │   └── ...
    │   ├── imgs_2/         # After images
    │   └── cm/
    │       └── cm.png      # Binary change mask (may be RGB/RGBA)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        patch_size: int = 256,
        bands: List[str] = None,
        augment: bool = True,
        normalize: bool = True,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.bands = bands or OSCD_BANDS
        self.augment = augment and split == "train"
        self.normalize = normalize

        # Get city list based on split — ONLY cities with labels (cm.png)
        if split == "train":
            candidates = OSCD_TRAIN_CITIES
        elif split in ("val", "test"):
            candidates = OSCD_VAL_CITIES
        else:
            candidates = OSCD_TRAIN_CITIES + OSCD_VAL_CITIES

        # Filter: city must exist AND have cm.png label
        self.cities = [
            c for c in candidates
            if (self.root_dir / c).exists() and (self.root_dir / c / "cm" / "cm.png").exists()
        ]

        if not self.cities:
            logger.warning(f"No OSCD cities found in {root_dir}. Using mock data.")
            self.use_mock = True
            self.cities = ["mock_city"]
        else:
            self.use_mock = False

        # Cache for per-city reference dimensions (from the highest-res band)
        self._ref_dims = {}

        # Build index of patches
        self.patches = self._build_patch_index()
        logger.info(f"OSCD {split}: {len(self.cities)} cities, {len(self.patches)} patches")

    def _get_reference_dims(self, city: str) -> Tuple[int, int]:
        """Get the reference (highest-resolution) image dimensions for a city."""
        if city in self._ref_dims:
            return self._ref_dims[city]

        city_dir = self.root_dir / city / "imgs_1"
        # Use the first available 10m band as reference (B02, B03, B04, B08)
        ref_bands = ["B02", "B03", "B04", "B08"] + self.bands
        h, w = 512, 512  # fallback

        for bname in ref_bands:
            bfile = _find_band_file(city_dir, bname)
            if bfile is not None and HAS_RASTERIO:
                with rasterio.open(str(bfile)) as src:
                    h, w = src.height, src.width
                break

        self._ref_dims[city] = (h, w)
        return h, w

    def _build_patch_index(self) -> List[dict]:
        """Create a list of (city, row, col) patch coordinates."""
        patches = []

        if self.use_mock:
            for i in range(100):
                patches.append({
                    "city": "mock_city",
                    "row": (i // 10) * self.patch_size,
                    "col": (i % 10) * self.patch_size,
                })
            return patches

        for city in self.cities:
            h, w = self._get_reference_dims(city)

            # Use overlapping patches (50% stride) for training to get more samples
            # Non-overlapping for validation to avoid inflated metrics
            stride = self.patch_size // 2 if self.split == "train" else self.patch_size
            for row in range(0, h - self.patch_size + 1, stride):
                for col in range(0, w - self.patch_size + 1, stride):
                    patches.append({
                        "city": city,
                        "row": row,
                        "col": col,
                        "img_h": h,
                        "img_w": w,
                    })

        return patches

    def _load_bands(self, city: str, time_dir: str) -> np.ndarray:
        """
        Load requested bands for a city at one time period.
        All bands are resampled to the reference (10m) resolution.
        """
        city_dir = self.root_dir / city / time_dir
        ref_h, ref_w = self._get_reference_dims(city)
        band_arrays = []

        for band_name in self.bands:
            band_file = _find_band_file(city_dir, band_name)

            if band_file is not None and HAS_RASTERIO:
                with rasterio.open(str(band_file)) as src:
                    data = src.read(1).astype(np.float32)

                # Resample to reference resolution if needed
                if data.shape != (ref_h, ref_w):
                    if HAS_SCIPY:
                        zoom_h = ref_h / data.shape[0]
                        zoom_w = ref_w / data.shape[1]
                        data = scipy_zoom(data, (zoom_h, zoom_w), order=1)
                        # Ensure exact dimensions after zoom rounding
                        data = data[:ref_h, :ref_w]
                        if data.shape[0] < ref_h or data.shape[1] < ref_w:
                            data = np.pad(data, (
                                (0, ref_h - data.shape[0]),
                                (0, ref_w - data.shape[1]),
                            ))
                    else:
                        # Simple nearest-neighbor resize without scipy
                        row_idx = np.linspace(0, data.shape[0] - 1, ref_h).astype(int)
                        col_idx = np.linspace(0, data.shape[1] - 1, ref_w).astype(int)
                        data = data[np.ix_(row_idx, col_idx)]

                band_arrays.append(data)
            else:
                # Band file missing — fill with zeros
                logger.debug(f"Band {band_name} not found for {city}/{time_dir}, using zeros")
                band_arrays.append(np.zeros((ref_h, ref_w), dtype=np.float32))

        return np.stack(band_arrays, axis=0)  # (C, H, W)

    def _load_change_mask(self, city: str) -> np.ndarray:
        """Load binary change mask for a city, resampled to reference dims."""
        city_dir = self.root_dir / city
        cm_file = city_dir / "cm" / "cm.png"
        ref_h, ref_w = self._get_reference_dims(city)

        if cm_file.exists():
            from PIL import Image
            mask_img = Image.open(str(cm_file))
            mask = np.array(mask_img)

            # Handle multi-channel masks (RGB or RGBA)
            if mask.ndim == 3:
                # Use first channel (or max across channels) for binarization
                mask = mask[:, :, 0]

            # Binarize: pixel > 128 means "change"
            mask = (mask > 128).astype(np.float32)

            # Resize mask to match reference band dimensions
            if mask.shape != (ref_h, ref_w):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((ref_w, ref_h), PILImage.NEAREST)
                mask = (np.array(mask_pil) > 128).astype(np.float32)
        else:
            # Mock mask — random sparse changes
            mask = np.zeros((ref_h, ref_w), dtype=np.float32)
            np.random.seed(hash(city) % 2**32)
            for _ in range(np.random.randint(2, 8)):
                cy, cx = np.random.randint(50, min(ref_h, ref_w) - 50, 2)
                radius = np.random.randint(10, 40)
                yy, xx = np.mgrid[:ref_h, :ref_w]
                circle = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius ** 2
                mask[circle] = 1.0

        return mask

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            img_before: (C, patch_size, patch_size) float32
            img_after:  (C, patch_size, patch_size) float32
            mask:       (patch_size, patch_size) long (0 or 1)
        """
        patch_info = self.patches[idx]
        city = patch_info["city"]
        row = patch_info["row"]
        col = patch_info["col"]

        if self.use_mock:
            return self._get_mock_item()

        # Load full images for this city
        img_before = self._load_bands(city, "imgs_1")
        img_after = self._load_bands(city, "imgs_2")
        mask = self._load_change_mask(city)

        # Crop patch
        r_end = min(row + self.patch_size, img_before.shape[1])
        c_end = min(col + self.patch_size, img_before.shape[2])
        img_before = img_before[:, row:r_end, col:c_end]
        img_after = img_after[:, row:r_end, col:c_end]
        mask = mask[row:r_end, col:c_end]

        # Pad to exact patch_size if at edges
        _, ph, pw = img_before.shape
        if ph < self.patch_size or pw < self.patch_size:
            pad_h = self.patch_size - ph
            pad_w = self.patch_size - pw
            img_before = np.pad(img_before, ((0, 0), (0, pad_h), (0, pad_w)))
            img_after = np.pad(img_after, ((0, 0), (0, pad_h), (0, pad_w)))
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)))

        # Normalize each band to [0, 1]
        if self.normalize:
            for c in range(img_before.shape[0]):
                for img in [img_before, img_after]:
                    bmin, bmax = img[c].min(), img[c].max()
                    if bmax > bmin:
                        img[c] = (img[c] - bmin) / (bmax - bmin)

        # Augmentation
        if self.augment:
            img_before, img_after, mask = self._augment(img_before, img_after, mask)

        return (
            torch.from_numpy(img_before.copy()).float(),
            torch.from_numpy(img_after.copy()).float(),
            torch.from_numpy(mask.copy()).long(),
        )

    def _augment(
        self, img_a: np.ndarray, img_b: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random augmentations consistently to both images and mask."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img_a = np.flip(img_a, axis=2).copy()
            img_b = np.flip(img_b, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            img_a = np.flip(img_a, axis=1).copy()
            img_b = np.flip(img_b, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random 90° rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img_a = np.rot90(img_a, k, axes=(1, 2)).copy()
            img_b = np.rot90(img_b, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        return img_a, img_b, mask

    def _get_mock_item(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a mock training sample."""
        C = len(self.bands)
        H = W = self.patch_size

        img_a = torch.rand(C, H, W)
        img_b = img_a.clone()

        # Add some changes to the second image
        mask = torch.zeros(H, W, dtype=torch.long)
        cy, cx = H // 2, W // 2
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        circle = ((yy - cy) ** 2 + (xx - cx) ** 2) < 30 ** 2
        mask[circle] = 1
        img_b[:, circle] += torch.rand(C, circle.sum()) * 0.3

        return img_a, img_b, mask
