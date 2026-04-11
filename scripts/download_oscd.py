"""
OSCD dataset downloader — Onera Satellite Change Detection dataset.

HOW TO DOWNLOAD (IEEE DataPort — manual):
==========================================
1. Log in to: https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection
2. Download these two ZIP files:
     - Onera Satellite Change Detection dataset - Images.zip   (~4.2 GB)
     - Onera Satellite Change Detection dataset - Train Labels.zip (~12 MB)
3. Extract BOTH zips into:  data/oscd/
4. You should end up with this structure:
     data/oscd/
     ├── aguasclaras/
     │   ├── imgs_1/          ← before images (per-band .tif files)
     │   │   ├── B01.tif
     │   │   ├── B02.tif
     │   │   └── ...
     │   ├── imgs_2/          ← after images
     │   └── cm/
     │       └── cm.png       ← binary change mask
     ├── bercy/
     ├── bordeaux/
     ├── paris/
     └── ... (14 city folders total)

NOTE: The test split cities (brasilia, montpellier, etc.) do NOT have ground-truth
      change masks in the public release. Only train cities are labelled.

After extracting:  python scripts/download_oscd.py  (to verify the structure)
Then train:        python scripts/train_backend.py
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OSCD_DIR = ROOT / "data" / "oscd"

OSCD_TRAIN_CITIES = [
    "aguasclaras", "bercy", "bordeaux", "nantes", "paris", "rennes",
    "saclay_e", "abudhabi", "cupertino", "pisa", "beihai",
    "hongkong", "beirut", "mumbai",
]
OSCD_TEST_CITIES = [
    "brasilia", "montpellier", "norcia", "rio", "saclay_w",
    "valencia", "dubai", "lasvegas", "milano", "chongqing",
]

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


def verify_structure():
    """Check what's in data/oscd/ and report what's found / missing."""
    logger.info(f"Checking OSCD structure in: {OSCD_DIR}")
    OSCD_DIR.mkdir(parents=True, exist_ok=True)

    found_train = []
    found_test  = []
    issues      = []

    for city in OSCD_TRAIN_CITIES:
        city_dir = OSCD_DIR / city
        if not city_dir.exists():
            continue

        imgs1 = city_dir / "imgs_1"
        imgs2 = city_dir / "imgs_2"
        cm    = city_dir / "cm" / "cm.png"

        if not imgs1.exists() or not imgs2.exists():
            issues.append(f"  {city}: missing imgs_1/ or imgs_2/")
            continue
        if not cm.exists():
            issues.append(f"  {city}: missing cm/cm.png (change mask)")
            continue

        # Count band files
        n_bands1 = len(list(imgs1.glob("*.tif")))
        n_bands2 = len(list(imgs2.glob("*.tif")))
        if n_bands1 == 0 or n_bands2 == 0:
            issues.append(f"  {city}: no .tif band files found")
            continue

        found_train.append(city)

    for city in OSCD_TEST_CITIES:
        city_dir = OSCD_DIR / city
        if city_dir.exists():
            found_test.append(city)

    # Report
    logger.info(f"\n{'='*55}")
    logger.info(f"  OSCD STRUCTURE REPORT")
    logger.info(f"{'='*55}")
    logger.info(f"  Train cities found (with labels): {len(found_train)}/14")
    for c in found_train:
        logger.info(f"    ✓ {c}")

    if found_test:
        logger.info(f"\n  Test cities found (no labels): {len(found_test)}/10")
        for c in found_test:
            logger.info(f"    ✓ {c}")

    if issues:
        logger.warning(f"\n  Issues:")
        for issue in issues:
            logger.warning(issue)

    missing_train = [c for c in OSCD_TRAIN_CITIES if c not in found_train]
    if missing_train:
        logger.info(f"\n  Missing train cities: {missing_train}")

    if len(found_train) == 0:
        logger.error("\n  ✗ NO train data found!")
        logger.info("\n  → Please download from IEEE DataPort:")
        logger.info("    https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection")
        logger.info(f"\n  → Extract both ZIPs into:  {OSCD_DIR}")
        return False
    else:
        logger.info(f"\n  ✓ Ready to train with {len(found_train)} city/cities.")
        logger.info(f"  → Run: python scripts/train_backend.py")
        return True


def try_torchgeo_download():
    """Attempt TorchGeo auto-download (works only with proper credentials)."""
    try:
        from torchgeo.datasets import OSCD
        logger.info("Attempting TorchGeo auto-download (requires dataset credentials)...")
        dataset = OSCD(root=str(OSCD_DIR), split="train", download=True)
        logger.info(f"TorchGeo OSCD downloaded: {len(dataset)} samples")
        return True
    except ImportError:
        logger.warning("torchgeo not installed, skipping auto-download.")
        return False
    except Exception as e:
        logger.warning(f"TorchGeo auto-download failed: {e}")
        logger.info("This is expected — OSCD requires a manual download from IEEE DataPort.")
        return False


if __name__ == "__main__":
    logger.info("Trying TorchGeo auto-download first...")
    success = try_torchgeo_download()

    if not success:
        logger.info("\nFalling back to manual verification...")

    verify_structure()
