"""
Fetch all image URLs from Supabase and print them in a clean format.
Gets: before_rgb, after_rgb, change_mask (combined = all 3 types) for up to 16 scans.
Run: python scripts/fetch_image_urls.py
"""
import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)

from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ SUPABASE_URL or SUPABASE_KEY missing in .env")
    sys.exit(1)

client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("🛰️  Fetching scan image URLs from Supabase...\n")
print("=" * 70)

result = client.table("scans").select(
    "scan_id, city, timestamp, before_rgb_url, after_rgb_url, change_mask_url, overlay_url, report_pdf_url"
).order("timestamp", desc=True).limit(16).execute()

scans = result.data
if not scans:
    print("❌ No scans found in the database. Run a scan first!")
    sys.exit(1)

print(f"✅ Found {len(scans)} scans. Showing URLs for all:\n")

# --- For PPT / WhatsApp sharing ---
all_before = []
all_after  = []
all_mask   = []
all_pdf    = []

for i, s in enumerate(scans, 1):
    city      = s.get("city", "Unknown")
    ts        = s.get("timestamp", "N/A")[:19]
    before    = s.get("before_rgb_url", "")
    after     = s.get("after_rgb_url",  "")
    mask      = s.get("change_mask_url","")
    overlay   = s.get("overlay_url",    "")
    pdf       = s.get("report_pdf_url", "")

    print(f"┌── [{i:02d}] {city}  |  {ts}")
    print(f"│   BEFORE  : {before or '—'}")
    print(f"│   AFTER   : {after  or '—'}")
    print(f"│   MASK    : {mask   or '—'}")
    print(f"│   OVERLAY : {overlay or '—'}")
    print(f"└   PDF     : {pdf    or '—'}\n")

    if before: all_before.append(before)
    if after:  all_after.append(after)
    if mask:   all_mask.append(mask)
    if pdf:    all_pdf.append(pdf)

# --- Combined URL dump for easy copy-paste into PPT ---
print("=" * 70)
print("\n📋 COMBINED URL LIST (copy-paste ready for PPT / WhatsApp):\n")

print("🟡 BEFORE IMAGES:")
for u in all_before:
    print(f"  {u}")

print("\n🟢 AFTER IMAGES:")
for u in all_after:
    print(f"  {u}")

print("\n🔴 CHANGE MASKS:")
for u in all_mask:
    print(f"  {u}")

print("\n📄 PDF REPORTS:")
for u in all_pdf:
    print(f"  {u}")

# --- Save to JSON as well ---
output = {
    "total_scans": len(scans),
    "before_images": all_before,
    "after_images": all_after,
    "change_masks": all_mask,
    "pdf_reports": all_pdf,
    "all_scans_detail": [
        {
            "index": i + 1,
            "city": s.get("city"),
            "timestamp": s.get("timestamp"),
            "before": s.get("before_rgb_url"),
            "after": s.get("after_rgb_url"),
            "mask": s.get("change_mask_url"),
            "overlay": s.get("overlay_url"),
            "pdf": s.get("report_pdf_url"),
        }
        for i, s in enumerate(scans)
    ]
}

out_file = PROJECT_ROOT / "scan_image_urls.json"
with open(out_file, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ All URLs also saved to: {out_file}")
print(f"   Total Before: {len(all_before)} | After: {len(all_after)} | Masks: {len(all_mask)} | PDFs: {len(all_pdf)}")
