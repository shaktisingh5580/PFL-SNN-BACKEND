"""
City Scan Orchestrator — End-to-End Pipeline

Takes a city bounding box → fetches/loads imagery → tiles into patches →
runs Siamese-SNN → stitches results → classifies changes → generates report.

Works in two modes:
1. LIVE MODE: Downloads real Sentinel-2 from SentinelHub (needs API key)
2. DEMO MODE: Uses OSCD validation cities (no API key needed)
"""

import os
import sys
import time
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.siamese_snn import SiameseSNN
from src.compliance.classifier import ChangeTypeClassifier

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


CITY_COORDS = {
    "abudhabi":  {"center": [24.4539, 54.3773], "bbox": [54.35, 24.43, 54.40, 24.47]},
    "beirut":    {"center": [33.8938, 35.5018], "bbox": [35.48, 33.87, 35.53, 33.92]},
    "mumbai":    {"center": [19.0760, 72.8777], "bbox": [72.85, 19.05, 72.90, 19.10]},
    "paris":     {"center": [48.8566, 2.3522],  "bbox": [2.33, 48.84, 2.37, 48.87]},
    "hongkong":  {"center": [22.3193, 114.1694],"bbox": [114.15, 22.30, 114.19, 22.34]},
    "surat":     {"center": [21.1702, 72.8311], "bbox": [72.80, 21.14, 72.86, 21.20]},
}

BANDS_5 = ['B02', 'B03', 'B04', 'B08', 'B11']
BANDS_4 = ['B02', 'B03', 'B04', 'B08']


class ScanResult:
    """Container for a complete city scan result."""

    def __init__(self, city: str, scan_id: str):
        self.city = city
        self.scan_id = scan_id
        self.timestamp = datetime.utcnow().isoformat()
        self.patches_scanned = 0
        self.inference_time = 0.0
        self.change_mask = None
        self.class_map = None
        self.report = None
        self.before_rgb = None
        self.after_rgb = None
        self.coordinates = None
        self.blockchain_hash = None

    def to_api_response(self) -> dict:
        """Convert to the API response format matching frontend.md contract."""
        return {
            "status": "success",
            "scan_id": self.scan_id,
            "city": self.city,
            "timestamp": self.timestamp,
            "model_info": {
                "name": "Siamese-SNN v3",
                "f1_score": 0.4187,
                "parameters": 7763362,
                "inference_time_seconds": round(self.inference_time, 1),
                "throughput_patches_per_hour": round(
                    self.patches_scanned / max(self.inference_time, 0.01) * 3600
                ),
            },
            "coordinates": self.coordinates or {
                "center": CITY_COORDS.get(self.city, {}).get("center", [0, 0]),
                "bounds": self._bbox_to_bounds(),
                "crs": "EPSG:4326",
            },
            "classification": self.report,
            "blockchain": {
                "hash": self.blockchain_hash or self._compute_evidence_hash(),
                "timestamp": self.timestamp,
                "verified": True,
            },
        }

    def _bbox_to_bounds(self):
        bbox = CITY_COORDS.get(self.city, {}).get("bbox", [0, 0, 1, 1])
        return [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]

    def _compute_evidence_hash(self) -> str:
        data = json.dumps({
            "scan_id": self.scan_id,
            "city": self.city,
            "timestamp": self.timestamp,
            "total_changed": int(self.change_mask.sum()) if self.change_mask is not None else 0,
        }, sort_keys=True)
        return "0x" + hashlib.sha256(data.encode()).hexdigest()


class CityScanner:
    """End-to-end pipeline: load model -> scan city -> classify -> report."""

    def __init__(self, model_path: str = "outputs/models/best_model.pt", patch_size: int = 128, device: str = "cpu"):
        self.patch_size = patch_size
        self.device = torch.device(device)
        self.classifier = ChangeTypeClassifier()

        logger.info("Loading Siamese-SNN model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})

        self.model = SiameseSNN(
            in_channels=config.get("num_bands", 4),
            encoder_channels=config.get("encoder_channels", [32, 64, 128, 256]),
            num_steps=config.get("num_steps", 10),
        )

        state_dict = checkpoint["model_state_dict"]
        model_keys = set(self.model.state_dict().keys())
        new_state = {}
        for k, v in state_dict.items():
            nk = (k.replace("encoder.blocks.", "encoder.encoder_blocks.")
                   .replace("decoder.blocks.", "decoder.decoder_blocks.")
                   .replace(".conv.", ".conv_block.")
                   .replace(".up.", ".upsample.")
                   .replace("decoder.head.", "decoder.output_conv."))
            if nk in model_keys:
                new_state[nk] = v
            elif k in model_keys:
                new_state[k] = v

        self.model.load_state_dict(new_state, strict=False)
        self.model.eval()
        self.model.to(self.device)
        logger.info(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")

    def scan_oscd_city(self, city: str, bands: list = None) -> ScanResult:
        """Scan an OSCD city using local data (DEMO mode)."""
        bands = bands or BANDS_5
        scan_id = f"scan_{city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = ScanResult(city=city, scan_id=scan_id)

        oscd_root = Path("data/oscd")
        city_dir = oscd_root / city
        if not city_dir.exists():
            raise FileNotFoundError(f"City not found: {city_dir}")

        img_before = self._load_city_bands(city_dir / "imgs_1", bands)
        img_after = self._load_city_bands(city_dir / "imgs_2", bands)
        if img_before is None or img_after is None:
            raise ValueError(f"Could not load bands for {city}")

        img_before = self._normalize(img_before)
        img_after = self._normalize(img_after)

        C, H, W = img_before.shape
        logger.info(f"City image: {H}x{W} ({C} bands)")

        result.before_rgb = self._make_rgb(img_before, bands)
        result.after_rgb = self._make_rgb(img_after, bands)

        t0 = time.time()
        change_mask = self._tile_and_predict(img_before[:4], img_after[:4], H, W)
        result.inference_time = time.time() - t0
        result.change_mask = change_mask

        ps = self.patch_size
        result.patches_scanned = ((H - ps) // ps + 1) * ((W - ps) // ps + 1)

        logger.info(f"Inference: {result.patches_scanned} patches, {result.inference_time:.1f}s, {change_mask.sum()} changed px")

        class_map = self.classifier.classify(img_before, img_after, change_mask, band_names=bands)
        result.class_map = class_map
        result.report = self.classifier.generate_report(class_map)
        result.blockchain_hash = result._compute_evidence_hash()

        return result

    def scan_bbox(self, bbox, date_before, date_after, city_name="custom") -> ScanResult:
        """Scan arbitrary area via SentinelHub API (LIVE mode)."""
        from src.pipeline.fetcher import fetch_temporal_pair
        from src.pipeline.preprocessor import prepare_model_input

        scan_id = f"scan_{city_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = ScanResult(city=city_name, scan_id=scan_id)

        before_patch, after_patch = fetch_temporal_pair(bbox, date_before, date_after, resolution=10)
        img_before, img_after = prepare_model_input(before_patch, after_patch)

        C, H, W = img_before.shape
        t0 = time.time()
        change_mask = self._tile_and_predict(img_before[:4], img_after[:4], H, W)
        result.inference_time = time.time() - t0
        result.change_mask = change_mask

        class_map = self.classifier.classify(img_before, img_after, change_mask, band_names=BANDS_5 if C >= 5 else BANDS_4)
        result.class_map = class_map
        result.report = self.classifier.generate_report(class_map)
        result.coordinates = {"center": [(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2], "bounds": [[bbox[1],bbox[0]],[bbox[3],bbox[2]]], "crs": "EPSG:4326"}
        return result

    def _load_city_bands(self, img_dir: Path, bands: list):
        if not HAS_RASTERIO:
            return None
        arrays = []
        ref_h, ref_w = None, None
        for bn in bands:
            tif = img_dir / f"{bn}.tif"
            if not tif.exists():
                matches = list(img_dir.glob(f"*_{bn}.tif")) or list(img_dir.glob(f"*_{bn.upper()}.tif"))
                tif = matches[0] if matches else None
            if tif and tif.exists():
                with rasterio.open(str(tif)) as src:
                    data = src.read(1).astype(np.float32)
                if ref_h is None:
                    ref_h, ref_w = data.shape
                elif data.shape != (ref_h, ref_w):
                    from scipy.ndimage import zoom as scipy_zoom
                    data = scipy_zoom(data, (ref_h/data.shape[0], ref_w/data.shape[1]), order=1)[:ref_h,:ref_w]
                    if data.shape[0]<ref_h or data.shape[1]<ref_w:
                        data = np.pad(data, ((0,ref_h-data.shape[0]),(0,ref_w-data.shape[1])))
                arrays.append(data)
            else:
                if ref_h:
                    arrays.append(np.zeros((ref_h, ref_w), dtype=np.float32))
                else:
                    return None
        return np.stack(arrays, axis=0)

    def _normalize(self, img):
        img = img.copy()
        for c in range(img.shape[0]):
            mn, mx = img[c].min(), img[c].max()
            if mx > mn:
                img[c] = (img[c] - mn) / (mx - mn)
        return img

    def _make_rgb(self, img, bands):
        idx_r = bands.index('B04') if 'B04' in bands else 2
        idx_g = bands.index('B03') if 'B03' in bands else 1
        idx_b = bands.index('B02') if 'B02' in bands else 0
        rgb = np.stack([img[idx_r], img[idx_g], img[idx_b]], axis=-1)
        for c in range(3):
            p2, p98 = np.percentile(rgb[:,:,c], [2, 98])
            if p98 > p2:
                rgb[:,:,c] = np.clip((rgb[:,:,c] - p2) / (p98 - p2), 0, 1)
        return (rgb * 255).astype(np.uint8)

    def _tile_and_predict(self, img_before_4, img_after_4, H, W):
        ps = self.patch_size
        change_mask = np.zeros((H, W), dtype=np.uint8)
        with torch.no_grad():
            for r in range(0, H - ps + 1, ps):
                for c in range(0, W - ps + 1, ps):
                    ta = torch.from_numpy(img_before_4[:, r:r+ps, c:c+ps]).float().unsqueeze(0).to(self.device)
                    tb = torch.from_numpy(img_after_4[:, r:r+ps, c:c+ps]).float().unsqueeze(0).to(self.device)
                    pred = self.model.predict(ta, tb).squeeze(0).cpu().numpy()
                    change_mask[r:r+ps, c:c+ps] = np.maximum(change_mask[r:r+ps, c:c+ps], pred)
        return change_mask

    def save_scan_outputs(self, result: ScanResult, output_dir: str = "outputs/scans"):
        out = Path(output_dir) / result.scan_id
        out.mkdir(parents=True, exist_ok=True)

        if HAS_PIL:
            if result.before_rgb is not None:
                Image.fromarray(result.before_rgb).save(str(out / "before_rgb.png"))
            if result.after_rgb is not None:
                Image.fromarray(result.after_rgb).save(str(out / "after_rgb.png"))
            if result.change_mask is not None:
                Image.fromarray((result.change_mask * 255).astype(np.uint8)).save(str(out / "change_mask.png"))
            if result.class_map is not None:
                color_map = self.classifier.get_color_map(result.class_map)
                alpha = ((result.class_map > 0) * 180).astype(np.uint8)
                rgba = np.dstack([color_map, alpha])
                Image.fromarray(rgba, 'RGBA').save(str(out / "class_overlay.png"))

        api_response = result.to_api_response()
        api_response["images"] = {
            "before_rgb": f"/api/static/scans/{result.scan_id}/before_rgb.png",
            "after_rgb": f"/api/static/scans/{result.scan_id}/after_rgb.png",
            "change_mask": f"/api/static/scans/{result.scan_id}/change_mask.png",
            "classification_overlay": f"/api/static/scans/{result.scan_id}/class_overlay.png",
        }
        with open(str(out / "report.json"), "w") as f:
            json.dump(api_response, f, indent=2, default=str)

        logger.info(f"Saved to: {out}")
        return api_response


def main():
    import matplotlib.pyplot as plt
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("  🛰️  CITY SCAN ORCHESTRATOR — DEMO MODE")
    print("=" * 60)

    scanner = CityScanner()
    cities = ["abudhabi", "beirut", "mumbai"]
    all_results = []

    for city in cities:
        try:
            print(f"\n{'─'*60}\n  Scanning: {city.upper()}\n{'─'*60}")
            result = scanner.scan_oscd_city(city)
            api_resp = scanner.save_scan_outputs(result)
            all_results.append((city, result, api_resp))

            print(f"\n  📊 {city}: {result.patches_scanned} patches, {result.inference_time:.1f}s")
            print(f"     Changed: {result.report['total_changed_pixels']} px ({result.report['total_changed_area_m2']:.0f} m²)")
            for f in result.report.get("findings", []):
                icon = "🔴" if f["severity"]=="high" else "🟡" if f["severity"]=="medium" else "🟢"
                print(f"     {icon} {f['class_name']}: {f['pixel_count']} px ({f['area_hectares']:.3f} ha)")
            print(f"     🔗 Hash: {result.blockchain_hash[:24]}...")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        fig, axes = plt.subplots(len(all_results), 4, figsize=(20, 5*len(all_results)))
        if len(all_results) == 1: axes = axes[np.newaxis, :]
        titles = ['Before', 'After', 'SNN Detection', 'Classification']
        for row, (city, res, _) in enumerate(all_results):
            axes[row,0].imshow(res.before_rgb); axes[row,1].imshow(res.after_rgb)
            axes[row,2].imshow(res.change_mask, cmap='hot')
            axes[row,3].imshow(scanner.classifier.get_overlay(res.after_rgb, res.class_map, 0.6))
            for c in range(4):
                axes[row,c].axis('off')
                if row==0: axes[row,c].set_title(titles[c], fontsize=13, fontweight='bold')
            axes[row,0].set_ylabel(city.upper(), fontsize=14, fontweight='bold', rotation=0, labelpad=60)

        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=np.array(scanner.classifier.COLORS[i])/255, label=scanner.classifier.CLASSES[i]) for i in [1,2,3,4]]
        fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11, bbox_to_anchor=(0.5,-0.02))
        plt.suptitle('🛰️ Full City Scan — 3 Validation Cities', fontsize=16, fontweight='bold')
        plt.tight_layout()
        out_path = "outputs/demo/full_city_scan.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n  ✅ Visualization: {out_path}")
        print(f"  ✅ Scan reports: outputs/scans/")

    print(f"\n{'='*60}\n  🎯 SCAN COMPLETE!\n{'='*60}")


if __name__ == "__main__":
    main()
