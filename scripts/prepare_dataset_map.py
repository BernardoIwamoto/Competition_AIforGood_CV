
import os
import pandas as pd
from pathlib import Path

def main():
    raw_dir = Path("data/raw_gid")
    label_dir = raw_dir / "GID-label"
    
    print(" scanning files...")
    
    # 1. Index all labels
    # Labels are expected to be in GID-label (flat)
    # We map "stem" -> "path"
    label_map = {}
    if label_dir.exists():
        for p in label_dir.glob("*"):
            if p.suffix.lower() in ['.png', '.tif', '.tiff', '.jpg']:
                label_map[p.stem] = p
    
    print(f"Found {len(label_map)} labels in {label_dir}")

    # 2. Find all images
    # Images are in GID-img-* folders, potentially nested
    image_paths = []
    for img_folder in raw_dir.glob("GID-img-*"):
        if img_folder.is_dir():
            # Recursive search for images
            for ext in ['*.tif', '*.tiff', '*.jpg', '*.png', '*.jpeg']:
                image_paths.extend(list(img_folder.rglob(ext)))
            
    print(f"Found {len(image_paths)} images in GID-img-* directories.")
    
    # 3. Match
    data = []
    matched_count = 0
    
    for img_path in image_paths:
        stem = img_path.stem
        
        # Easy match: exact stem match
        if stem in label_map:
            data.append({
                "image_path": str(img_path),
                "label_path": str(label_map[stem])
            })
            matched_count += 1
        
    df = pd.DataFrame(data)
    print(f"Matched {len(df)} pairs.")
    
    if len(df) > 0:
        out_csv = Path("data/dataset_map.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")
    else:
        print("No matches found. Check directory structure and filenames.")
        # Debug info if no matches
        if len(image_paths) > 0:
            print(f"Sample image stem: {image_paths[0].stem}")
        if len(label_map) > 0:
            print(f"Sample label stem: {list(label_map.keys())[0]}")

if __name__ == "__main__":
    main()
