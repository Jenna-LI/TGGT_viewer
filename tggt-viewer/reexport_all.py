#!/usr/bin/env python3
"""Re-export all objects with fixed RGB/depth ordering."""
import json
import os
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path("/home/chuanruo/TGGT_viewer/tggt-viewer/results")
CHECKPOINT = "/home/chuanruo/vggt_train/training/logs/exp197/ckpts/checkpoint_190.pt"
EPOCH = 190
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"

# Seen objects use these frames
SEEN_FRAMES = "3,5,7,9,11,12,34,45,56,67,77,78,89,90"

def get_objects_to_export():
    """Get list of objects with their data paths."""
    objects = []
    for obj_dir in sorted(RESULTS_DIR.iterdir()):
        if not obj_dir.is_dir():
            continue
        runs_file = obj_dir / "runs.json"
        if not runs_file.exists():
            continue

        with open(runs_file) as f:
            runs = json.load(f)

        data_path = runs.get("data_path")
        if not data_path or not os.path.exists(data_path):
            print(f"Skipping {obj_dir.name}: no valid data_path")
            continue

        # Determine if it's a seen or unseen object based on experiment name
        experiments = runs.get("experiments", [])
        is_unseen = "my_data_32" in experiments

        objects.append({
            "id": obj_dir.name,
            "data_path": data_path,
            "is_unseen": is_unseen
        })

    return objects

def export_object(obj):
    """Export a single object."""
    obj_id = obj["id"]
    data_path = obj["data_path"]
    is_unseen = obj["is_unseen"]

    cmd = [
        "python", "export_to_viewer.py",
        "--data", data_path,
        "--checkpoint", CHECKPOINT,
        "--output", "results",
        "--epoch", str(EPOCH),
        "--device", DEVICE
    ]

    # Add frames filter for seen objects
    if not is_unseen:
        cmd.extend(["--frames", SEEN_FRAMES])
        cmd.append("--val")

    print(f"\n{'='*60}")
    print(f"Exporting: {obj_id}")
    print(f"Data path: {data_path}")
    print(f"Unseen: {is_unseen}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"SUCCESS: {obj_id}")
        return True
    else:
        print(f"FAILED: {obj_id}")
        return False

def main():
    objects = get_objects_to_export()
    print(f"Found {len(objects)} objects to export")
    print(f"Device: {DEVICE}")

    success = 0
    failed = 0

    for i, obj in enumerate(objects):
        print(f"\n[{i+1}/{len(objects)}]", end="")
        if export_object(obj):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"Success: {success}, Failed: {failed}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
