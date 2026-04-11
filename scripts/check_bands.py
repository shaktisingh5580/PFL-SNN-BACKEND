import rasterio
from pathlib import Path
from PIL import Image
import numpy as np

cities = ["aguasclaras", "bercy", "bordeaux", "paris", "mumbai"]
bands = ["B02", "B03", "B04", "B08"]

results = []
for city in cities:
    root = Path(f"data/oscd/{city}")
    if not root.exists():
        continue
    results.append(f"\n=== {city} ===")
    for b in bands:
        f = root / "imgs_1" / f"{b}.tif"
        if f.exists():
            with rasterio.open(str(f)) as src:
                results.append(f"  imgs_1/{b}.tif: {src.height}x{src.width}")
        else:
            results.append(f"  imgs_1/{b}.tif: MISSING")
    cm_path = root / "cm" / "cm.png"
    if cm_path.exists():
        mask = np.array(Image.open(str(cm_path)))
        results.append(f"  cm/cm.png: shape={mask.shape}, dtype={mask.dtype}")
    else:
        results.append(f"  cm/cm.png: MISSING")

with open("band_info.txt", "w") as f:
    f.write("\n".join(results))

print("Done. See band_info.txt")
