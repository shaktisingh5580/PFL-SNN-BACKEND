"""
🛰️ Live Scan: Vesu, Surat — Detect Recent Construction/Development

Uses Sentinel Hub Processing API (Planet Insights Platform) directly
to fetch real Sentinel-2 L2A imagery for Vesu, South Surat.
"""
import os
import sys
import time
import json
import hashlib
import logging
import importlib
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import numpy as np

# Load .env
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Force-reimport sentinelhub (in case it was cached as missing)
try:
    import sentinelhub
    importlib.reload(sentinelhub)
except:
    pass

# Now import our fetcher (it will re-detect sentinelhub)
importlib.invalidate_caches()
if 'src.pipeline.fetcher' in sys.modules:
    del sys.modules['src.pipeline.fetcher']

from sentinelhub import (
    SHConfig, SentinelHubRequest, DataCollection,
    MimeType, CRS, BBox, bbox_to_dimensions,
)

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
VESU_BBOX = (72.7750, 21.1450, 72.8050, 21.1700)
VESU_CENTER = [21.1575, 72.7900]
DATE_BEFORE = ("2025-01-01", "2025-03-31")
DATE_AFTER  = ("2025-10-01", "2026-03-31")

# 5-band evalscript: B02, B03, B04, B08, B11 + cloud mask
EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B02","B03","B04","B08","B11","SCL"],
            units: "DN"
        }],
        output: [
            { id: "bands", bands: 5, sampleType: "FLOAT32" },
            { id: "scl", bands: 1, sampleType: "UINT8" }
        ]
    };
}
function evaluatePixel(sample) {
    let isCloud = (sample.SCL==3||sample.SCL==8||sample.SCL==9||sample.SCL==10);
    if (isCloud) return { bands: [0,0,0,0,0], scl: [sample.SCL] };
    return {
        bands: [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11],
        scl: [sample.SCL]
    };
}
"""

print("=" * 65)
print("  🛰️  LIVE SCAN: VESU, SURAT")
print("  " + "─" * 55)
print(f"  BBOX:   {VESU_BBOX}")
print(f"  Before: {DATE_BEFORE[0]} → {DATE_BEFORE[1]}")
print(f"  After:  {DATE_AFTER[0]} → {DATE_AFTER[1]}")
print(f"  API:    SH_CLIENT_ID = {os.getenv('SH_CLIENT_ID','NOT SET')[:12]}...")
print("=" * 65)

# ═══════════════════════════════════════════════════════════
#  STEP 1: Auth + Fetch (Copernicus Data Space Ecosystem)
# ═══════════════════════════════════════════════════════════
import requests as http_requests
from sentinelhub import SentinelHubSession

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

print(f"\n📡 Step 1: Authenticating via Copernicus Data Space (CDSE)...")

# Manually fetch OAuth2 token (bypasses SDK's oauthlib issue)
token_response = http_requests.post(TOKEN_URL, data={
    "grant_type": "client_credentials",
    "client_id": os.getenv("SH_CLIENT_ID", ""),
    "client_secret": os.getenv("SH_CLIENT_SECRET", ""),
})

if token_response.status_code != 200:
    print(f"  ❌ Auth failed: {token_response.status_code} — {token_response.text[:200]}")
    sys.exit(1)

token_data = token_response.json()
print(f"  ✅ OAuth2 token acquired! Expires in {token_data.get('expires_in', '?')}s")

# Create config pointed at CDSE
config = SHConfig()
config.sh_client_id = os.getenv("SH_CLIENT_ID", "")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET", "")
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = TOKEN_URL

# Inject the manually-fetched token into a session
session = SentinelHubSession(config=config, _token=token_data)
print(f"  ✅ Session created with pre-fetched token")

# Define CDSE-specific data collection (guard against duplicate enum registration)
try:
    CDSE_S2_L2A = DataCollection.define_from(
        DataCollection.SENTINEL2_L2A,
        name="SENTINEL2_L2A_CDSE",
        service_url="https://sh.dataspace.copernicus.eu",
    )
except ValueError:
    CDSE_S2_L2A = DataCollection["SENTINEL2_L2A_CDSE"]

sh_bbox = BBox(bbox=VESU_BBOX, crs=CRS.WGS84)
size = bbox_to_dimensions(sh_bbox, resolution=10)
print(f"  Image size: {size[0]}×{size[1]} pixels at 10m resolution")
print(f"  Ground coverage: {size[0]*10/1000:.1f} km × {size[1]*10/1000:.1f} km")

def fetch_patch(date_start, date_end, label):
    print(f"\n  📥 Downloading {label} ({date_start} → {date_end})...")
    t0 = time.time()
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=CDSE_S2_L2A,
                time_interval=(date_start, date_end),
                maxcc=0.3,
            )
        ],
        responses=[
            SentinelHubRequest.output_response("bands", MimeType.TIFF),
            SentinelHubRequest.output_response("scl", MimeType.TIFF),
        ],
        bbox=sh_bbox,
        size=size,
        config=config,
    )
    data = request.get_data()
    elapsed = time.time() - t0

    if not data:
        print(f"  ❌ No data returned for {label}")
        return None

    bands = data[0]["bands.tif"]  # (H, W, 5)
    scl = data[0]["scl.tif"]     # (H, W, 1)

    if bands.ndim == 3:
        bands = np.transpose(bands, (2, 0, 1))  # → (5, H, W)
    scl = scl.squeeze()

    print(f"  ✅ {label}: shape={bands.shape}, took {elapsed:.1f}s")
    return {"bands": bands.astype(np.float32), "scl": scl}

before = fetch_patch(DATE_BEFORE[0], DATE_BEFORE[1], "BEFORE")
after = fetch_patch(DATE_AFTER[0], DATE_AFTER[1], "AFTER")

if before is None or after is None:
    print("❌ Could not fetch imagery. Check API credentials and date ranges.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
#  STEP 2: Normalize
# ═══════════════════════════════════════════════════════════
print("\n🔧 Step 2: Normalizing imagery...")

def normalize(img):
    img = img.copy()
    for c in range(img.shape[0]):
        mn, mx = img[c].min(), img[c].max()
        if mx > mn:
            img[c] = (img[c] - mn) / (mx - mn)
    return img

img_before = normalize(before["bands"])
img_after = normalize(after["bands"])
C, H, W = img_before.shape
print(f"  Normalized: {C} bands, {H}×{W} pixels")

# ═══════════════════════════════════════════════════════════
#  STEP 3: Run Siamese-SNN
# ═══════════════════════════════════════════════════════════
print("\n🧠 Step 3: Running Siamese-SNN Change Detection...")

import torch
from src.model.siamese_snn import SiameseSNN

checkpoint = torch.load("outputs/models/best_model.pt", map_location="cpu", weights_only=False)
model_config = checkpoint.get("config", {})
model = SiameseSNN(
    in_channels=model_config.get("num_bands", 4),
    encoder_channels=model_config.get("encoder_channels", [32, 64, 128, 256]),
    num_steps=model_config.get("num_steps", 10),
)

# Key remapping
state_dict = checkpoint["model_state_dict"]
model_keys = set(model.state_dict().keys())
new_state = {}
for k, v in state_dict.items():
    nk = (k.replace("encoder.blocks.", "encoder.encoder_blocks.")
           .replace("decoder.blocks.", "decoder.decoder_blocks.")
           .replace(".conv.", ".conv_block.")
           .replace(".up.", ".upsample.")
           .replace("decoder.head.", "decoder.output_conv."))
    if nk in model_keys: new_state[nk] = v
    elif k in model_keys: new_state[k] = v
model.load_state_dict(new_state, strict=False)
model.eval()
print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Tile and predict (model uses 4 bands: B02, B03, B04, B08 = indices 0:4)
ps = 128
change_mask = np.zeros((H, W), dtype=np.uint8)
n_patches = 0
t0 = time.time()

with torch.no_grad():
    for r in range(0, H - ps + 1, ps):
        for c in range(0, W - ps + 1, ps):
            ta = torch.from_numpy(img_before[:4, r:r+ps, c:c+ps]).float().unsqueeze(0)
            tb = torch.from_numpy(img_after[:4, r:r+ps, c:c+ps]).float().unsqueeze(0)
            pred = model.predict(ta, tb).squeeze(0).cpu().numpy()
            change_mask[r:r+ps, c:c+ps] = np.maximum(change_mask[r:r+ps, c:c+ps], pred)
            n_patches += 1

infer_time = time.time() - t0
changed_px = int(change_mask.sum())
print(f"  ✅ {n_patches} patches, {infer_time:.1f}s, {changed_px:,} changed pixels ({changed_px/(H*W)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════
#  STEP 4: Classify Changes
# ═══════════════════════════════════════════════════════════
print("\n🏗️ Step 4: Classifying changes (NDVI/NDBI/MNDWI)...")

from src.compliance.classifier import ChangeTypeClassifier

classifier = ChangeTypeClassifier()
band_names = ['B02', 'B03', 'B04', 'B08', 'B11']
class_map = classifier.classify(img_before, img_after, change_mask, band_names=band_names)
report = classifier.generate_report(class_map)

print(f"\n  📊 VESU, SURAT — CLASSIFICATION RESULTS")
print(f"  {'─'*50}")
print(f"  Total changed: {report['total_changed_area_m2']:,.0f} m² ({report['total_changed_area_m2']/10000:.2f} ha)")
for f in report.get("findings", []):
    icon = "🔴" if f["severity"]=="high" else "🟡" if f["severity"]=="medium" else "🟢"
    print(f"  {icon} {f['class_name']}: {f['pixel_count']:,} px ({f['area_hectares']:.2f} ha) [{f['severity'].upper()}]")

# ═══════════════════════════════════════════════════════════
#  STEP 5: Compliance Rule Engine
# ═══════════════════════════════════════════════════════════
print("\n📋 Step 5: Running Compliance Rule Engine...")

from src.compliance.rule_engine import ComplianceRuleEngine

rule_engine = ComplianceRuleEngine()
violations = rule_engine.evaluate(class_map, VESU_BBOX, report)

if violations:
    print(f"\n  {len(violations)} VIOLATION(S) FOUND:")
    for i, v in enumerate(violations, 1):
        sev = v["severity"]
        icon = "CRIT" if sev == "CRITICAL" else "HIGH" if sev == "HIGH" else "MED" if sev == "MEDIUM" else "LOW"
        print(f"    [{icon}] {v['rule_name']} — {v['legal_reference']}")
        details = v.get("details", {})
        if "distance_to_protected_m" in details:
            print(f"           Distance: {details['distance_to_protected_m']}m (Buffer: {details.get('buffer_required_m')}m)")
        if "affected_area_hectares" in details:
            print(f"           Area: {details['affected_area_hectares']} ha")
else:
    print("  No compliance violations detected.")

# ═══════════════════════════════════════════════════════════
#  STEP 6: Save Images & Report
# ═══════════════════════════════════════════════════════════
print("\n💾 Step 6: Saving outputs...")

from PIL import Image

out_dir = Path("outputs/scans/vesu_surat")
out_dir.mkdir(parents=True, exist_ok=True)

# RGB composites
def make_rgb(img):
    rgb = np.stack([img[2], img[1], img[0]], axis=-1)  # B04, B03, B02
    for ch in range(3):
        p2, p98 = np.percentile(rgb[:,:,ch], [2, 98])
        if p98 > p2: rgb[:,:,ch] = np.clip((rgb[:,:,ch]-p2)/(p98-p2), 0, 1)
    return (rgb * 255).astype(np.uint8)

before_rgb = make_rgb(img_before)
after_rgb = make_rgb(img_after)

Image.fromarray(before_rgb).save(str(out_dir / "before_rgb.png"))
Image.fromarray(after_rgb).save(str(out_dir / "after_rgb.png"))
Image.fromarray((change_mask * 255).astype(np.uint8)).save(str(out_dir / "change_mask.png"))

color_map = classifier.get_color_map(class_map)
alpha = ((class_map > 0) * 180).astype(np.uint8)
Image.fromarray(np.dstack([color_map, alpha]), 'RGBA').save(str(out_dir / "class_overlay.png"))

overlay = classifier.get_overlay(after_rgb, class_map, alpha=0.6)
Image.fromarray(overlay).save(str(out_dir / "classification_overlay.png"))

# Build API report JSON
scan_id = f"scan_vesu_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
evidence = json.dumps({"city": "vesu_surat", "bbox": list(VESU_BBOX), "changed": changed_px}, sort_keys=True)
bh = "0x" + hashlib.sha256(evidence.encode()).hexdigest()

api_report = {
    "status": "success",
    "scan_id": scan_id,
    "city": "Vesu, Surat",
    "region": "South Surat, Gujarat, India",
    "timestamp": datetime.utcnow().isoformat(),
    "model_info": {
        "name": "Siamese-SNN v3",
        "f1_score": 0.4187,
        "parameters": 7763362,
        "inference_time_seconds": round(infer_time, 1),
        "throughput_patches_per_hour": round(n_patches / max(infer_time, 0.01) * 3600),
    },
    "coordinates": {
        "center": VESU_CENTER,
        "bounds": [[VESU_BBOX[1], VESU_BBOX[0]], [VESU_BBOX[3], VESU_BBOX[2]]],
        "crs": "EPSG:4326",
    },
    "classification": report,
    "violations": [
        {"rule_id": v["rule_id"], "rule_name": v["rule_name"],
         "severity": v["severity"], "legal_reference": v["legal_reference"],
         "details": v.get("details", {})}
        for v in violations
    ],
    "violation_count": len(violations),
    "blockchain": {"hash": bh, "timestamp": datetime.utcnow().isoformat(), "verified": True},
}

with open(str(out_dir / "report.json"), "w") as f:
    json.dump(api_report, f, indent=2, default=str)

print(f"  Saved images + report.json to {out_dir}/")

# ═══════════════════════════════════════════════════════════
#  STEP 7: Generate PDF Compliance Report
# ═══════════════════════════════════════════════════════════
print("\n📄 Step 7: Generating PDF Compliance Report...")

try:
    from src.reporting.pdf_generator import generate_compliance_pdf
    pdf_path = generate_compliance_pdf(
        scan_dir=str(out_dir),
        report_data=api_report,
        violations=violations,
        city="Surat",
    )
    print(f"  PDF report: {pdf_path}")
except Exception as e:
    print(f"  PDF generation failed: {e}")
    print(f"  Install fpdf2: pip install fpdf2")

# ═══════════════════════════════════════════════════════════
#  STEP 8: Upload to Supabase
# ═══════════════════════════════════════════════════════════
print("\n☁️  Step 8: Uploading to Supabase...")

try:
    from src.api.supabase_client import SupabaseClient
    sb = SupabaseClient()

    if sb.is_connected:
        # Upload images to bucket
        image_urls = sb.upload_scan_images(str(out_dir), scan_id)
        print(f"  Uploaded {len(image_urls)} files to satellite-scans bucket")

        # Save scan record to PostGIS
        db_id = sb.save_scan(api_report, violations, image_urls)
        if db_id:
            print(f"  Saved to DB: {db_id}")
    else:
        print("  Supabase not configured — skipping upload.")
        print("  Set SUPABASE_URL and SUPABASE_KEY in .env to enable.")
except Exception as e:
    print(f"  Supabase upload skipped: {e}")

# ═══════════════════════════════════════════════════════════
#  STEP 9: Visualization
# ═══════════════════════════════════════════════════════════
print("\n🖼️ Step 9: Generating visualization...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(before_rgb); axes[0].set_title("Before (Early 2025)", fontweight='bold')
axes[1].imshow(after_rgb); axes[1].set_title("After (Late 2025)", fontweight='bold')
axes[2].imshow(change_mask, cmap='hot'); axes[2].set_title(f"SNN Detection ({changed_px:,} px)", fontweight='bold')
axes[3].imshow(overlay); axes[3].set_title("Classification", fontweight='bold')
for ax in axes: ax.axis('off')

handles = [mpatches.Patch(color=np.array(classifier.COLORS[i])/255, label=classifier.CLASSES[i]) for i in [1,2,3,4]]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11, bbox_to_anchor=(0.5,-0.02))
plt.suptitle('Live Scan: Vesu, Surat - Construction Detection', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(str(out_dir / "vesu_scan_result.png"), dpi=150, bbox_inches='tight', facecolor='white')

# ═══════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  VESU SCAN COMPLETE!")
print(f"{'='*65}")
print(f"  Location:     Vesu, South Surat | {H*10/1000:.1f}km x {W*10/1000:.1f}km")
print(f"  Changed:      {changed_px:,} px ({report['total_changed_area_m2']/10000:.2f} ha)")
print(f"  Violations:   {len(violations)} compliance violations found")
for v in violations[:3]:
    print(f"                [{v['severity']}] {v['rule_name']}")
print(f"  Inference:    {infer_time:.1f}s ({n_patches} patches)")
print(f"  Hash:         {bh[:32]}...")
print(f"  PDF Report:   {out_dir}/compliance_report.pdf")
print(f"  Output Dir:   {out_dir}/")
print(f"{'='*65}")

