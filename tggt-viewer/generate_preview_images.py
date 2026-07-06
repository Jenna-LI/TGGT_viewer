#!/usr/bin/env python3
"""Generate static preview images from point clouds."""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_ply(path):
    """Read PLY file and return positions and colors."""
    with open(path, 'rb') as f:
        line = f.readline().decode().strip()
        if line != 'ply':
            raise ValueError("Not a PLY file")

        vertex_count = 0
        header_end = False

        while not header_end:
            line = f.readline().decode().strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if line == 'end_header':
                header_end = True

        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
        data = np.frombuffer(f.read(), dtype=dtype, count=vertex_count)

    pos = np.column_stack([data['x'], data['y'], data['z']])
    col = np.column_stack([data['r'], data['g'], data['b']]) / 255.0
    return pos, col

def render_pointcloud(pos, col, output_path):
    """Render point cloud to image using 2D projection."""
    # Use all points for density
    n_points = len(pos)

    # Center the point cloud
    center = pos.mean(axis=0)
    pos = pos - center

    # Rotate to get a good view
    theta_y = np.radians(135)
    theta_x = np.radians(20)

    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    pos = pos @ Ry.T @ Rx.T

    # Sort by depth for proper occlusion (back to front)
    depth_order = np.argsort(pos[:, 2])
    pos = pos[depth_order]
    col = col[depth_order]

    # Project to 2D (orthographic)
    x = pos[:, 0]
    y = pos[:, 1]

    # Normalize to image coordinates with padding
    x_range = x.max() - x.min() if x.max() > x.min() else 1
    y_range = y.max() - y.min() if y.max() > y.min() else 1
    scale = max(x_range, y_range) * 1.1

    x = (x - x.mean()) / scale + 0.5
    y = (y - y.mean()) / scale + 0.5

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')

    # Adaptive point size based on number of points
    # More points = smaller size, fewer points = larger size
    point_size = max(0.5, min(20, 50000 / n_points))

    # Plot all points
    ax.scatter(x, y, c=col, s=point_size, marker='o', edgecolors='none', alpha=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=100, facecolor='#0f0f23', pad_inches=0)
    plt.close()

def main():
    results_dir = Path("results")
    count = 0

    for obj_dir in sorted(results_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        output_img = obj_dir / "preview.png"

        # Find the full pointmap PLY
        ply_file = None
        for epoch_dir in obj_dir.iterdir():
            if epoch_dir.is_dir():
                candidate = epoch_dir / "points_pointmap.ply"
                if candidate.exists():
                    ply_file = candidate
                    break

        if not ply_file:
            continue

        try:
            pos, col = read_ply(ply_file)
            render_pointcloud(pos, col, output_img)
            count += 1
            print(f"[{count}] {obj_dir.name} ({len(pos)} pts)")
        except Exception as e:
            print(f"Error {obj_dir.name}: {e}")

    print(f"\nGenerated {count} previews.")

if __name__ == "__main__":
    main()
