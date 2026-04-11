"""Quick test: Load the trained model and run inference on a sample city."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

print("="*60)
print("  Loading trained Siamese-SNN model...")
print("="*60)

checkpoint = torch.load("outputs/models/best_model.pt", map_location="cpu", weights_only=False)
print(f"  Epoch: {checkpoint.get('epoch', '?')}")
print(f"  Best F1: {checkpoint.get('best_f1', '?'):.4f}")

config = checkpoint.get('config', {})
print(f"  Config: {config}")

# Build model
from src.model.siamese_snn import SiameseSNN
model = SiameseSNN(
    in_channels=config.get('num_bands', 4),
    encoder_channels=config.get('encoder_channels', [32, 64, 128, 256]),
    num_steps=config.get('num_steps', 10),
)

# Remap checkpoint keys to match local model structure
state_dict = checkpoint['model_state_dict']
model_state = model.state_dict()
model_keys = set(model_state.keys())
new_state = {}

for k, v in state_dict.items():
    new_key = k
    # Colab: "encoder.blocks" → local: "encoder.encoder_blocks"
    new_key = new_key.replace("encoder.blocks.", "encoder.encoder_blocks.")
    # Colab: "decoder.blocks" → local: "decoder.decoder_blocks"
    new_key = new_key.replace("decoder.blocks.", "decoder.decoder_blocks.")
    # Colab: ".conv." → local: ".conv_block."
    new_key = new_key.replace(".conv.", ".conv_block.")
    # Colab: ".up." → local: ".upsample."
    new_key = new_key.replace(".up.", ".upsample.")
    # Colab: "decoder.head." → local: "decoder.output_conv."
    new_key = new_key.replace("decoder.head.", "decoder.output_conv.")
    
    if new_key in model_keys:
        new_state[new_key] = v
    elif k in model_keys:
        new_state[k] = v

missing, unexpected = model.load_state_dict(new_state, strict=False)
loaded = len(model.state_dict()) - len(missing)
print(f"  Loaded {loaded}/{len(model.state_dict())} weight tensors")
if missing:
    # Filter out snntorch internal params (they'll use defaults)
    real_missing = [k for k in missing if not any(x in k for x in ['beta', 'threshold', 'graded_spikes', 'reset_mechanism'])]
    if real_missing:
        print(f"  ⚠️ Missing non-SNN keys: {real_missing[:5]}")
    else:
        print(f"  ℹ️ Only missing snntorch internal params ({len(missing)}) — using defaults")

model.eval()
print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

# Test on validation data
from src.training.dataset import OSCDDataset
val_ds = OSCDDataset(
    "data/oscd", split="val", patch_size=128,
    bands=['B02','B03','B04','B08'], augment=False
)
print(f"\n  Val dataset: {len(val_ds)} patches from {val_ds.cities}")

print("\n" + "="*60)
print("  Running inference...")
print("="*60)

tp = fp = fn = 0
t0 = time.time()
num_test = min(len(val_ds), 50)

with torch.no_grad():
    for i in range(num_test):
        img_a, img_b, mask = val_ds[i]
        img_a = img_a.unsqueeze(0)
        img_b = img_b.unsqueeze(0)
        pred = model.predict(img_a, img_b)
        pred = pred.squeeze(0)
        tp += ((pred == 1) & (mask == 1)).sum().item()
        fp += ((pred == 1) & (mask == 0)).sum().item()
        fn += ((pred == 0) & (mask == 1)).sum().item()
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{num_test} patches... (TP={tp} FP={fp} FN={fn})")

elapsed = time.time() - t0
prec = tp / (tp + fp + 1e-8)
rec = tp / (tp + fn + 1e-8)
f1 = 2 * prec * rec / (prec + rec + 1e-8)
iou = tp / (tp + fp + fn + 1e-8)

print(f"\n{'='*60}")
print(f"  ✅ INFERENCE RESULTS ({num_test} patches)")
print(f"{'='*60}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  IoU:       {iou:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  Time:      {elapsed:.1f}s ({elapsed/num_test:.2f}s per patch)")
print(f"  Throughput: {num_test/elapsed*3600:.0f} patches/hour")
print(f"{'='*60}")
