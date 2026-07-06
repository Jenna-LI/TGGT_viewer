#!/usr/bin/env python3
"""Add source_image paths to all cameras.json files."""
import json
import os
from pathlib import Path

RESULTS_DIR = Path("/home/chuanruo/TGGT_viewer/tggt-viewer/results")

def add_source_paths(obj_dir):
    runs_file = obj_dir / "runs.json"
    if not runs_file.exists():
        return False, "No runs.json"

    with open(runs_file) as f:
        runs = json.load(f)

    data_path = runs.get("data_path")
    if not data_path or not os.path.exists(data_path):
        return False, "Invalid data_path"

    # Find images directory
    source_dir = Path(data_path)
    images_subdir = None
    for p in source_dir.rglob("images"):
        if p.is_dir():
            images_subdir = p
            break

    if not images_subdir:
        return False, "No images subdir"

    updated_any = False
    for exp_dir in obj_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        cameras_file = exp_dir / "cameras.json"
        if not cameras_file.exists():
            continue

        with open(cameras_file) as f:
            cameras = json.load(f)

        # Update pred_cameras
        for cam in cameras.get("pred_cameras", []):
            view_id = cam.get("view_id", "")
            frame_idx = int(view_id[1:]) if view_id and view_id[0] in "tv" else 0
            src_path = images_subdir / f"frame_{frame_idx:05d}.png"
            if not src_path.exists():
                src_path = images_subdir / f"frame_{frame_idx:05d}.jpg"
            cam["source_image"] = str(src_path) if src_path.exists() else None

        # Update gt_cameras too
        for cam in cameras.get("gt_cameras", []):
            view_id = cam.get("view_id", "")
            frame_idx = int(view_id[1:]) if view_id and view_id[0] in "tv" else 0
            src_path = images_subdir / f"frame_{frame_idx:05d}.png"
            if not src_path.exists():
                src_path = images_subdir / f"frame_{frame_idx:05d}.jpg"
            cam["source_image"] = str(src_path) if src_path.exists() else None

        with open(cameras_file, "w") as f:
            json.dump(cameras, f, indent=2)

        updated_any = True

    return updated_any, "OK"

def main():
    objects = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(objects)} objects", flush=True)

    for i, obj_dir in enumerate(objects):
        print(f"[{i+1}/{len(objects)}] {obj_dir.name}...", flush=True)
        ok, msg = add_source_paths(obj_dir)
        if not ok:
            print(f"  SKIP: {msg}", flush=True)

    print("Done!", flush=True)

if __name__ == "__main__":
    main()
