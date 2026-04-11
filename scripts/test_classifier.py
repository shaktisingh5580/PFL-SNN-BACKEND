"""
Test the Change Type Classifier on validation cities.
Produces a multi-panel visualization showing Before, After, SNN Detection, and Classification.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.training.dataset import OSCDDataset
from src.model.siamese_snn import SiameseSNN
from src.compliance.classifier import ChangeTypeClassifier

# ═══════════════════════════════════════════════════════════════
#  1. Load Model
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  Loading Siamese-SNN Model...")
print("=" * 60)

checkpoint = torch.load("outputs/models/best_model.pt", map_location="cpu", weights_only=False)
config = checkpoint.get('config', {})

model = SiameseSNN(
    in_channels=config.get('num_bands', 4),
    encoder_channels=config.get('encoder_channels', [32, 64, 128, 256]),
    num_steps=config.get('num_steps', 10),
)

# Remap keys (Colab → local)
state_dict = checkpoint['model_state_dict']
model_keys = set(model.state_dict().keys())
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
model.load_state_dict(new_state, strict=False)
model.eval()
print(f"  ✅ Model loaded (F1={checkpoint.get('best_f1',0):.4f}, epoch {checkpoint.get('epoch','?')})")

# ═══════════════════════════════════════════════════════════════
#  2. Load Dataset — use 128 patch size (matches training!)
#     Load 5 bands for classification: B02, B03, B04, B08, B11
# ═══════════════════════════════════════════════════════════════
print("\n  Loading validation data (5 bands)...")

BANDS_5 = ['B02', 'B03', 'B04', 'B08', 'B11']
val_ds = OSCDDataset(
    "data/oscd", split="val", patch_size=128,
    bands=BANDS_5, augment=False
)
print(f"  ✅ {len(val_ds)} patches from {val_ds.cities}")

# ═══════════════════════════════════════════════════════════════
#  3. Find Best Patches (ones with most change)
# ═══════════════════════════════════════════════════════════════
print("\n  Scanning for patches with highest change...")
patch_scores = []
for i in range(len(val_ds)):
    _, _, mask = val_ds[i]
    change_pct = mask.sum().item() / mask.numel() * 100
    if change_pct > 0.5:  # At least 0.5% change
        patch_scores.append((i, change_pct))

patch_scores.sort(key=lambda x: -x[1])
best_patches = patch_scores[:6]  # Top 6
print(f"  ✅ Found {len(patch_scores)} patches with change, using top {len(best_patches)}")
for idx, pct in best_patches:
    print(f"     Patch {idx}: {pct:.1f}% change pixels")

# ═══════════════════════════════════════════════════════════════
#  4. Run Classification
# ═══════════════════════════════════════════════════════════════
classifier = ChangeTypeClassifier()

print("\n" + "=" * 60)
print("  Running Classification Pipeline...")
print("=" * 60)

results = []
for patch_idx, change_pct in best_patches:
    img_a, img_b, true_mask = val_ds[patch_idx]

    # Model uses 4 bands (B02, B03, B04, B08) → channels 0:4
    img_a_4band = img_a[:4].unsqueeze(0)
    img_b_4band = img_b[:4].unsqueeze(0)

    with torch.no_grad():
        pred_mask = model.predict(img_a_4band, img_b_4band).squeeze(0).numpy()

    # Classify using all 5 bands
    class_map = classifier.classify(
        img_a.numpy(), img_b.numpy(), pred_mask,
        band_names=BANDS_5
    )

    report = classifier.generate_report(class_map)
    results.append({
        'idx': patch_idx,
        'img_a': img_a.numpy(),
        'img_b': img_b.numpy(),
        'true_mask': true_mask.numpy(),
        'pred_mask': pred_mask,
        'class_map': class_map,
        'report': report,
    })

    # Print report
    print(f"\n  Patch {patch_idx} — {report['total_changed_pixels']} changed pixels "
          f"({report['total_changed_area_m2']:.0f} m²)")
    for f in report['findings']:
        print(f"    {'🔴' if f['severity']=='high' else '🟡' if f['severity']=='medium' else '🟢'} "
              f"{f['class_name']}: {f['pixel_count']} px ({f['area_hectares']:.3f} ha) "
              f"[{f['severity'].upper()}]")

# ═══════════════════════════════════════════════════════════════
#  5. Generate Visualization
# ═══════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

num_rows = min(len(results), 4)
fig, axes = plt.subplots(num_rows, 5, figsize=(25, 5 * num_rows))
if num_rows == 1:
    axes = axes[np.newaxis, :]

col_titles = ['Before (RGB)', 'After (RGB)', 'Ground Truth', 'SNN Detection', 'Change Classification']

for row, r in enumerate(results[:num_rows]):
    # RGB composite: B04=Red(idx2), B03=Green(idx1), B02=Blue(idx0)
    rgb_before = np.stack([r['img_a'][2], r['img_a'][1], r['img_a'][0]], axis=-1)
    rgb_after = np.stack([r['img_b'][2], r['img_b'][1], r['img_b'][0]], axis=-1)

    # Normalize for display (percentile stretch)
    for img in [rgb_before, rgb_after]:
        for c in range(3):
            p2, p98 = np.percentile(img[:, :, c], [2, 98])
            if p98 > p2:
                img[:, :, c] = np.clip((img[:, :, c] - p2) / (p98 - p2), 0, 1)

    # Overlay classification on After image
    color_map = classifier.get_color_map(r['class_map'])
    overlay = classifier.get_overlay(rgb_after, r['class_map'], alpha=0.6)

    axes[row, 0].imshow(rgb_before)
    axes[row, 1].imshow(rgb_after)
    axes[row, 2].imshow(r['true_mask'], cmap='Reds', vmin=0, vmax=1)
    axes[row, 3].imshow(r['pred_mask'], cmap='hot', vmin=0, vmax=1)
    axes[row, 4].imshow(overlay)

    for col in range(5):
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title(col_titles[col], fontsize=13, fontweight='bold')

    # Add city label
    patch_info = val_ds.patches[r['idx']]
    axes[row, 0].text(5, 15, f"{patch_info['city']}", color='white',
                      fontsize=10, fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Add legend
handles = []
for cid, name in classifier.CLASSES.items():
    if cid == 0: continue
    color = np.array(classifier.COLORS[cid]) / 255.0
    sev = classifier.SEVERITY[cid]
    handles.append(mpatches.Patch(color=color, label=f"{name} [{sev.upper()}]"))

fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11,
           frameon=True, fancybox=True, shadow=True,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle('🛰️ Siamese-SNN Change Detection + Type Classification',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()

os.makedirs("outputs/demo", exist_ok=True)
out_path = "outputs/demo/classification_test.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n  ✅ Saved: {out_path}")

# ═══════════════════════════════════════════════════════════════
#  6. Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  📊 CLASSIFICATION SUMMARY")
print("=" * 60)
total_by_class = {cid: 0 for cid in classifier.CLASSES if cid > 0}
for r in results:
    for cid in total_by_class:
        total_by_class[cid] += int(np.sum(r['class_map'] == cid))

for cid, count in total_by_class.items():
    if count > 0:
        area_ha = count * 100 / 10000  # 10m pixels
        print(f"  {classifier.CLASSES[cid]}: {count} pixels ({area_ha:.3f} ha)")

print(f"\n  ✅ Pipeline complete! Check: {out_path}")
