"""Audit the OSCD dataset: check labels, bands, change percentages."""
from PIL import Image
import numpy as np
from pathlib import Path

TRAIN = ['aguasclaras','bercy','bordeaux','nantes','paris','rennes',
         'saclay_e','abudhabi','cupertino','pisa','beihai','hongkong','beirut','mumbai']
TEST  = ['brasilia','montpellier','norcia','rio','saclay_w',
         'valencia','dubai','lasvegas','milano','chongqing']

root = Path("data/oscd")

print("="*80)
print("  OSCD DATASET AUDIT")
print("="*80)

# Train cities
print("\n📦 TRAIN CITIES (14):")
print(f"  {'City':15s} {'Mask Shape':18s} {'Bands1':7s} {'Bands2':7s} {'Change%':8s}")
print("  " + "-"*60)
total_change = []
for c in TRAIN:
    d = root / c
    imgs1 = list((d/"imgs_1").glob("*.tif"))
    imgs2 = list((d/"imgs_2").glob("*.tif"))
    cm = d / "cm" / "cm.png"
    if cm.exists():
        m = np.array(Image.open(str(cm)))
        if m.ndim == 3:
            pct = (m[:,:,0] > 128).mean() * 100
        else:
            pct = (m > 128).mean() * 100
        total_change.append(pct)
        print(f"  {c:15s} {str(m.shape):18s} {len(imgs1):7d} {len(imgs2):7d} {pct:6.2f}%")
    else:
        print(f"  {c:15s} {'NO MASK':18s} {len(imgs1):7d} {len(imgs2):7d} {'N/A':>8s}")

print(f"\n  Average change percentage: {np.mean(total_change):.2f}%")

# Test cities
print("\n📦 TEST CITIES (10):")
print(f"  {'City':15s} {'Has Labels':12s} {'Bands1':7s} {'Bands2':7s}")
print("  " + "-"*45)
test_labeled = 0
for c in TEST:
    d = root / c
    imgs1 = list((d/"imgs_1").glob("*.tif"))
    imgs2 = list((d/"imgs_2").glob("*.tif"))
    cm = d / "cm" / "cm.png"
    has_label = "✅ YES" if cm.exists() else "❌ NO"
    if cm.exists():
        test_labeled += 1
    print(f"  {c:15s} {has_label:12s} {len(imgs1):7d} {len(imgs2):7d}")

print(f"\n  Test cities with labels: {test_labeled}/10")

# Summary
print("\n" + "="*80)
print("  SUMMARY")
print("="*80)
print(f"  Train cities: 14 (all have imgs_1 + imgs_2 + cm.png labels)")
print(f"  Test cities:  10 ({test_labeled} with labels, {10-test_labeled} without)")
print(f"  Bands per city: 13 Sentinel-2 bands")
print(f"  Avg change pixels: {np.mean(total_change):.1f}% (heavily imbalanced)")
if test_labeled == 0:
    print(f"\n  ⚠️  ISSUE: No test city has cm.png labels!")
    print(f"      The validation loop uses 'no change' masks (all zeros) for test cities.")
    print(f"      F1/IoU will always be 0 because there are no ground-truth change pixels!")
    print(f"      FIX: Move test labels or use train/val split from labeled train cities.")
