#!/usr/bin/env python3
"""
Generate synthetic sample data for the 3D reconstruction viewer.
Creates point clouds, camera poses, metrics, and placeholder images.
"""

import json
import math
import os
import random
from pathlib import Path

# Configuration for sample objects
SAMPLES = {
    "sample_001": {
        "name": "Sample Object 01",
        "shape": "torus",  # torus shape
        "experiments": {
            "exp01": [5, 20, 50],
            "exp02": [5, 20]
        }
    },
    "sample_002": {
        "name": "Sample Object 02",
        "shape": "sphere",  # sphere shape
        "experiments": {
            "exp01": [5, 20]
        }
    }
}

NUM_CAMERAS = 12  # More cameras for meaningful train/val split
NUM_POINTS = 2000


def generate_sphere_points(n_points, noise_level=0.0, color_offset=0):
    """Generate points on a sphere with optional noise."""
    points = []
    for _ in range(n_points):
        # Random spherical coordinates
        theta = random.uniform(0, 2 * math.pi)
        phi = math.acos(2 * random.uniform(0, 1) - 1)

        r = 0.5 + random.gauss(0, noise_level)
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)

        # Color based on position (gradient)
        red = min(255, max(0, int(128 + 127 * x + color_offset)))
        green = min(255, max(0, int(128 + 127 * y + color_offset)))
        blue = min(255, max(0, int(128 + 127 * z)))

        points.append((x, y, z, red, green, blue))
    return points


def generate_torus_points(n_points, noise_level=0.0, color_offset=0):
    """Generate points on a torus with optional noise."""
    points = []
    R = 0.5  # major radius
    r = 0.2  # minor radius

    for _ in range(n_points):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, 2 * math.pi)

        # Add noise to the minor radius
        r_noise = r + random.gauss(0, noise_level)

        x = (R + r_noise * math.cos(phi)) * math.cos(theta)
        y = (R + r_noise * math.cos(phi)) * math.sin(theta)
        z = r_noise * math.sin(phi)

        # Color based on angle (rainbow gradient)
        hue = theta / (2 * math.pi)
        red = min(255, max(0, int(255 * (0.5 + 0.5 * math.sin(hue * 6.28)) + color_offset)))
        green = min(255, max(0, int(255 * (0.5 + 0.5 * math.sin(hue * 6.28 + 2.09)))))
        blue = min(255, max(0, int(255 * (0.5 + 0.5 * math.sin(hue * 6.28 + 4.19)))))

        points.append((x, y, z, red, green, blue))
    return points


def write_ply(filepath, points):
    """Write points to ASCII PLY file."""
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]} {p[4]} {p[5]}\n")


def generate_gt_cameras(n_cameras, val_ratio=0.3):
    """Generate ground truth camera poses arranged in a circle looking at origin."""
    cameras = []
    n_val = int(n_cameras * val_ratio)
    val_indices = set(random.sample(range(n_cameras), n_val))

    for i in range(n_cameras):
        angle = (2 * math.pi * i) / n_cameras
        distance = 2.0

        # Position on circle at height variation
        height = 0.5 * math.sin(angle * 2)
        x = distance * math.cos(angle)
        y = height
        z = distance * math.sin(angle)

        cameras.append({
            "view_id": f"{'v' if i in val_indices else 't'}{i}",
            "split": "val" if i in val_indices else "train",
            "position": [round(x, 4), round(y, 4), round(z, 4)],
            "look_at": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_deg": 50,
            "aspect": 1.333,
            "near": 0.01,
            "far": 0.3,
            "depth_image": f"depths/view_{i:03d}.png"
        })

    return cameras


def generate_pred_cameras(gt_cameras, epoch):
    """Generate predicted cameras with errors that decrease at higher epochs."""
    pred_cameras = []

    # Error magnitude decreases with epoch
    # At epoch 5: ~0.15 position error, at epoch 50: ~0.02 position error
    error_scale = 0.2 * (1 - epoch / 60)

    for gt_cam in gt_cameras:
        # Add random perturbation to position
        pos = gt_cam["position"]
        pred_pos = [
            round(pos[0] + random.gauss(0, error_scale), 4),
            round(pos[1] + random.gauss(0, error_scale * 0.5), 4),  # Less error in Y
            round(pos[2] + random.gauss(0, error_scale), 4)
        ]

        # Slightly perturb look_at as well
        look_at = gt_cam["look_at"]
        pred_look_at = [
            round(look_at[0] + random.gauss(0, error_scale * 0.1), 4),
            round(look_at[1] + random.gauss(0, error_scale * 0.1), 4),
            round(look_at[2] + random.gauss(0, error_scale * 0.1), 4)
        ]

        pred_cameras.append({
            "view_id": gt_cam["view_id"],
            "split": gt_cam["split"],  # Preserve train/val split
            "position": pred_pos,
            "look_at": pred_look_at,
            "up": gt_cam["up"],
            "fov_deg": gt_cam["fov_deg"],
            "aspect": gt_cam["aspect"],
            "near": gt_cam["near"],
            "far": gt_cam["far"],
            "depth_image": gt_cam["depth_image"]
        })

    return pred_cameras


def generate_metrics(experiment, epoch, base_chamfer=0.015, base_emd=0.2):
    """Generate synthetic metrics that improve with higher epochs."""
    # Metrics improve (decrease) with higher epochs
    # Different experiments have slightly different baselines
    exp_offset = 0.001 if experiment == "exp01" else 0.002

    # Improvement factor based on epoch (higher epoch = lower error)
    improvement = 1.0 - (epoch / 100) * 0.7
    noise = random.uniform(-0.001, 0.001)

    # Also add pose error metrics
    pose_error = 0.2 * improvement + random.uniform(-0.02, 0.02)

    return {
        "chamfer_distance": round((base_chamfer + exp_offset) * improvement + noise, 4),
        "emd": round((base_emd + exp_offset * 10) * improvement + noise * 10, 4),
        "surface_error_pct": round(((base_chamfer * 500) + exp_offset * 100) * improvement + noise * 50, 2),
        "pose_error_t": round(pose_error, 4),
        "pose_error_r": round(pose_error * 5, 2)  # Rotation error in degrees
    }


def create_gradient_png(filepath, color1, color2, width=200, height=150):
    """Create a simple gradient PNG image using raw bytes (no PIL needed)."""
    import struct
    import zlib

    def png_chunk(chunk_type, data):
        chunk_len = len(data)
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xffffffff
        return struct.pack('>I', chunk_len) + chunk + struct.pack('>I', crc)

    # Generate raw RGB data with gradient
    raw_data = []
    for y in range(height):
        row = [0]  # Filter byte (no filter)
        t = y / height
        for x in range(width):
            r = int(color1[0] * (1 - t) + color2[0] * t)
            g = int(color1[1] * (1 - t) + color2[1] * t)
            b = int(color1[2] * (1 - t) + color2[2] * t)
            row.extend([r, g, b])
        raw_data.append(bytes(row))

    raw_data = b''.join(raw_data)
    compressed = zlib.compress(raw_data, 9)

    # PNG signature
    png = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    png += png_chunk(b'IHDR', ihdr_data)

    # IDAT chunk
    png += png_chunk(b'IDAT', compressed)

    # IEND chunk
    png += png_chunk(b'IEND', b'')

    with open(filepath, 'wb') as f:
        f.write(png)


def create_cover_image(filepath, shape):
    """Create a cover image with neutral academic colors."""
    if shape == "torus":
        create_gradient_png(filepath, (240, 240, 245), (180, 185, 200))
    else:
        create_gradient_png(filepath, (235, 240, 250), (175, 190, 210))


def generate_sample_data():
    """Generate all sample data."""
    results_dir = Path("results")
    metrics_summary = []

    for sample_id, config in SAMPLES.items():
        sample_dir = results_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Create cover image
        create_cover_image(sample_dir / "gt_cover.png", config["shape"])
        print(f"Created cover image for {sample_id}")

        # Generate GT cameras (same across all runs)
        gt_cameras = generate_gt_cameras(NUM_CAMERAS)

        for experiment, epochs in config["experiments"].items():
            for epoch in epochs:
                run_name = f"{experiment}_epoch{epoch:03d}"
                run_dir = sample_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                # Create depths directory
                depths_dir = run_dir / "depths"
                depths_dir.mkdir(exist_ok=True)

                # Generate pred cameras with errors
                pred_cameras = generate_pred_cameras(gt_cameras, epoch)

                # Generate point clouds with noise based on epoch
                # Lower epoch = more noise
                noise_level = 0.1 * (1 - epoch / 60)

                if config["shape"] == "torus":
                    # Predicted point cloud (with noise)
                    pred_points = generate_torus_points(NUM_POINTS, noise_level)
                    # GT point cloud (clean)
                    gt_points = generate_torus_points(NUM_POINTS, 0.01)
                else:
                    pred_points = generate_sphere_points(NUM_POINTS, noise_level)
                    gt_points = generate_sphere_points(NUM_POINTS, 0.01)

                # Write predicted point cloud
                write_ply(run_dir / "points.ply", pred_points)
                # Write GT point cloud
                write_ply(run_dir / "points_gt.ply", gt_points)

                # Write cameras with both pred and GT
                cameras_data = {
                    "pred_cameras": pred_cameras,
                    "gt_cameras": gt_cameras
                }
                with open(run_dir / "cameras.json", 'w') as f:
                    json.dump(cameras_data, f, indent=2)

                # Generate and write metrics
                metrics = generate_metrics(experiment, epoch)
                with open(run_dir / "metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)

                # Add to summary
                metrics_summary.append({
                    "object_id": sample_id,
                    "experiment": experiment,
                    "epoch": epoch,
                    **metrics
                })

                # Create images directory
                images_dir = run_dir / "images"
                images_dir.mkdir(exist_ok=True)

                # Create depth and RGB images (gradient placeholders)
                for i in range(NUM_CAMERAS):
                    # Vary colors slightly per view
                    hue_offset = i / NUM_CAMERAS

                    # Depth image (blue-ish gradients)
                    c1 = (
                        int(50 + 100 * hue_offset),
                        int(50 + 50 * hue_offset),
                        int(100 + 100 * (1 - hue_offset))
                    )
                    c2 = (
                        int(150 + 50 * hue_offset),
                        int(100 + 100 * hue_offset),
                        int(50 + 150 * (1 - hue_offset))
                    )
                    create_gradient_png(depths_dir / f"view_{i:03d}.png", c1, c2)

                    # RGB image (warmer colors)
                    rgb_c1 = (
                        int(200 + 50 * hue_offset),
                        int(150 + 80 * hue_offset),
                        int(100 + 50 * (1 - hue_offset))
                    )
                    rgb_c2 = (
                        int(180 + 50 * hue_offset),
                        int(120 + 100 * hue_offset),
                        int(80 + 100 * (1 - hue_offset))
                    )
                    create_gradient_png(images_dir / f"view_{i:03d}.png", rgb_c1, rgb_c2)

                print(f"Generated {sample_id}/{run_name}")

    # Write metrics summary
    with open(results_dir / "metrics_summary.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nGenerated metrics_summary.json with {len(metrics_summary)} entries")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    generate_sample_data()
    print("\nDone! Sample data generated successfully.")
