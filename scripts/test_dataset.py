"""Quick test: load 1 patch from the OSCD dataset and print shapes."""
import sys, logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

from src.training.dataset import OSCDDataset

ds = OSCDDataset(str(ROOT / "data" / "oscd"), split="train", patch_size=128, bands=["B02","B03","B04","B08"])
print(f"Dataset size: {len(ds)} patches")
print(f"Cities: {ds.cities}")

# Load first and last patch
a, b, m = ds[0]
print(f"Patch 0: img_a={a.shape}, img_b={b.shape}, mask={m.shape}")

a2, b2, m2 = ds[len(ds)-1]
print(f"Patch {len(ds)-1}: img_a={a2.shape}, img_b={b2.shape}, mask={m2.shape}")

# Try loading 2 patches to simulate a batch
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))
print(f"Batch shapes: {batch[0].shape}, {batch[1].shape}, {batch[2].shape}")
print("ALL OK!")
