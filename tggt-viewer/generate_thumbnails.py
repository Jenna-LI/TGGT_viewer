#!/usr/bin/env python3
"""Generate small thumbnail point clouds for gallery preview."""

import os
import numpy as np
from pathlib import Path

def read_ply(path):
    """Read PLY file and return vertices and colors."""
    with open(path, 'rb') as f:
        # Read header
        line = f.readline().decode().strip()
        if line != 'ply':
            raise ValueError("Not a PLY file")

        vertex_count = 0
        has_color = False
        header_end = False

        while not header_end:
            line = f.readline().decode().strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if 'red' in line or 'green' in line or 'blue' in line:
                has_color = True
            if line == 'end_header':
                header_end = True

        # Read binary data
        if has_color:
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                     ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
        else:
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

        data = np.frombuffer(f.read(), dtype=dtype, count=vertex_count)

    return data, has_color

def write_ply(path, data, has_color):
    """Write PLY file."""
    with open(path, 'wb') as f:
        # Header
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {len(data)}\n'.encode())
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        if has_color:
            f.write(b'property uchar red\n')
            f.write(b'property uchar green\n')
            f.write(b'property uchar blue\n')
        f.write(b'end_header\n')
        f.write(data.tobytes())

def downsample(data, target_count=2000):
    """Randomly downsample to target count."""
    if len(data) <= target_count:
        return data
    indices = np.random.choice(len(data), target_count, replace=False)
    indices.sort()
    return data[indices]

def main():
    results_dir = Path(__file__).parent / "results"

    count = 0
    skipped = 0

    for obj_dir in sorted(results_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        for epoch_dir in obj_dir.iterdir():
            if not epoch_dir.is_dir():
                continue

            src = epoch_dir / "points_pointmap.ply"
            dst = epoch_dir / "points_thumb.ply"

            if not src.exists():
                continue

            if dst.exists():
                skipped += 1
                continue

            try:
                data, has_color = read_ply(src)
                small = downsample(data, 2000)
                write_ply(dst, small, has_color)
                count += 1
                print(f"[{count}] {obj_dir.name}: {len(data)} -> {len(small)} points")
            except Exception as e:
                print(f"Error processing {src}: {e}")

    print(f"\nDone! Generated {count} thumbnails, skipped {skipped} existing.")

if __name__ == "__main__":
    main()
