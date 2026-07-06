#!/usr/bin/env python3
"""Fix RGB images using source_image paths from cameras.json."""
import json
import os
from pathlib import Path
from PIL import Image

RESULTS_DIR = Path("/home/chuanruo/TGGT_viewer/tggt-viewer/results")

def fix_object_images(obj_dir):
    """Fix RGB images for a single object."""
    fixed_any = False

    for exp_dir in obj_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        cameras_file = exp_dir / "cameras.json"
        images_dir = exp_dir / "images"
        if not cameras_file.exists() or not images_dir.exists():
            continue

        with open(cameras_file) as f:
            cameras = json.load(f)

        count = 0
        for i, cam in enumerate(cameras.get("pred_cameras", [])):
            src_path = cam.get("source_image")
            if src_path and os.path.exists(src_path):
                dst_path = images_dir / f"view_{i:03d}.png"
                img = Image.open(src_path).convert("RGB")
                img = img.resize((224, 224), Image.LANCZOS)
                img.save(dst_path)
                count += 1

        if count > 0:
            print(f"  Fixed {count} images in {exp_dir.name}", flush=True)
            fixed_any = True

    return fixed_any

def main():
    objects = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(objects)} objects", flush=True)

    success = 0
    for i, obj_dir in enumerate(objects):
        print(f"[{i+1}/{len(objects)}] {obj_dir.name}...", flush=True)
        if fix_object_images(obj_dir):
            success += 1

    print(f"\nDone! Fixed: {success}", flush=True)

if __name__ == "__main__":
    main()
