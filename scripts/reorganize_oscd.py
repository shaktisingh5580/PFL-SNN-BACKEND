import os
import shutil
from pathlib import Path

def main():
    root = Path(__file__).parent.parent
    data_dir = root / "data"
    oscd_dir = data_dir / "oscd"
    oscd_dir.mkdir(parents=True, exist_ok=True)

    img_src = data_dir / "Onera Satellite Change Detection dataset - Images" / "Onera Satellite Change Detection dataset - Images"
    label_src = data_dir / "Onera Satellite Change Detection dataset - Train Labels" / "Onera Satellite Change Detection dataset - Train Labels"

    # Move images
    if img_src.exists():
        print(f"Moving images from {img_src}")
        for city_dir in img_src.iterdir():
            if city_dir.is_dir():
                city = city_dir.name
                dest_city = oscd_dir / city
                dest_city.mkdir(parents=True, exist_ok=True)
                
                for f in ["imgs_1", "imgs_2"]:
                    src_f = city_dir / f
                    if src_f.exists():
                        dest_f = dest_city / f
                        if not dest_f.exists():
                            shutil.move(str(src_f), str(dest_f))
                            print(f"  Moved {city}/{f}")

    # Move labels
    if label_src.exists():
        print(f"Moving labels from {label_src}")
        for city_dir in label_src.iterdir():
            if city_dir.is_dir():
                city = city_dir.name
                dest_city = oscd_dir / city
                dest_city.mkdir(parents=True, exist_ok=True)
                
                src_cm = city_dir / "cm"
                if src_cm.exists():
                    dest_cm = dest_city / "cm"
                    if not dest_cm.exists():
                        shutil.move(str(src_cm), str(dest_cm))
                        print(f"  Moved {city}/cm")

    print("Reorganization complete!")

if __name__ == "__main__":
    main()
