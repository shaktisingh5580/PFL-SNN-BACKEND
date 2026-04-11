"""
Test scan script — Sachin, Surat (no frontend needed).

Runs the full pipeline orchestrator directly and prints
every SSE event + debug output to the terminal.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from src.pipeline.orchestrator import run_pipeline

# ── Sachin, Surat ──
# Sachin GIDC is an industrial area — good for detecting construction changes.
# Center: ~21.089°N, 72.875°E
# Using same area size as Vesu scan: 0.030° x 0.025° (~3km x 2.7km, 315x272 px, 4 patches)
BBOX = (72.860, 21.0765, 72.890, 21.1015)
CITY = "Sachin, Surat"

print("=" * 60)
print(f"  SCANNING: {CITY}")
print(f"  BBOX: {BBOX}")
print(f"  Expected size: ~128x128 px at 10m resolution")
print("=" * 60)

for event_json in run_pipeline(
    bbox=BBOX,
    city=CITY,
    date_before=("2025-01-01", "2025-03-31"),
    date_after=("2025-10-01", "2026-03-31"),
    resolution=10,
):
    event = json.loads(event_json.strip())
    step = event.get("step", "?")
    status = event.get("status", "")
    msg = event.get("message", "")
    progress = event.get("progress", 0)
    data = event.get("data", {})

    icon = {"running": "⏳", "complete": "✅", "error": "❌",
            "started": "🚀", "skipped": "⏭️", "warning": "⚠️",
            "finished": "🏁"}.get(status, "•")

    print(f"\n  {icon} [{progress:3d}%] Step {step}: {msg}")

    # Print confidence stats if available
    if "confidence_stats" in data:
        cs = data["confidence_stats"]
        print(f"       ╰─ Confidence: min={cs['min']}, max={cs['max']}, mean={cs['mean']}, threshold={cs['threshold_used']}")

    # Print violations if available
    if "violations" in data and isinstance(data["violations"], list):
        for v in data["violations"]:
            print(f"       ╰─ {v}")

    # Print image URLs at the end
    if status == "finished":
        print("\n" + "=" * 60)
        print("  SCAN COMPLETE!")
        print(f"  Changed pixels: {data.get('changed_pixels', 0):,}")
        print(f"  Changed area:   {data.get('change_hectares', 0):.2f} ha")
        print(f"  Violations:     {data.get('violation_count', 0)}")
        print(f"  Evidence hash:  {data.get('evidence_hash', 'N/A')[:40]}...")
        print(f"\n  PDF:    {data.get('pdf_url', 'N/A')}")
        print(f"  Before: {data.get('before_url', 'N/A')}")
        print(f"  After:  {data.get('after_url', 'N/A')}")
        print(f"  Mask:   {data.get('mask_url', 'N/A')}")
        print("=" * 60)
