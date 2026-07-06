#!/usr/bin/env python3
"""
Recompute camera pose metrics (RRA, RTA, AUC) for existing exported data.

This script reads cameras.json to get pred/gt extrinsics and updates metrics.json
with the camera_pose metrics without needing to re-run inference.

Usage:
    python recompute_camera_pose_metrics.py --results_dir results
    python recompute_camera_pose_metrics.py --results_dir results --dry-run
"""

import os
import json
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np


def rotation_matrix_to_angle(R):
    """Convert rotation matrix to angle in degrees using Rodrigues formula."""
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle_rad)


def compute_relative_rotation_error(R_pred_i, R_pred_j, R_gt_i, R_gt_j):
    """
    Compute RRA (Relative Rotation Accuracy) between a pair of views.
    Returns error in degrees.
    """
    R_rel_pred = R_pred_j @ R_pred_i.T
    R_rel_gt = R_gt_j @ R_gt_i.T
    R_error = R_rel_pred @ R_rel_gt.T
    return rotation_matrix_to_angle(R_error)


def compute_relative_translation_error(t_pred_i, t_pred_j, t_gt_i, t_gt_j):
    """
    Compute RTA (Relative Translation Accuracy) between a pair of views.
    Returns error in degrees.
    """
    t_rel_pred = t_pred_j - t_pred_i
    t_rel_gt = t_gt_j - t_gt_i

    norm_pred = np.linalg.norm(t_rel_pred)
    norm_gt = np.linalg.norm(t_rel_gt)

    if norm_pred < 1e-8 or norm_gt < 1e-8:
        return 0.0

    t_rel_pred = t_rel_pred / norm_pred
    t_rel_gt = t_rel_gt / norm_gt

    cos_angle = np.clip(np.dot(t_rel_pred, t_rel_gt), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def compute_auc(errors, thresholds):
    """
    Compute Area Under the Curve for accuracy-vs-threshold.
    Returns dictionary mapping threshold to AUC value (normalized to [0, 1]).
    """
    if len(errors) == 0:
        return {t: 0.0 for t in thresholds}

    errors = np.array(errors)
    auc_results = {}

    for max_thresh in thresholds:
        sweep = np.linspace(0, max_thresh, 100)
        accuracies = np.array([np.mean(errors < tau) for tau in sweep])
        # Use trapezoidal integration (compatible with both old and new numpy)
        # Manual implementation: sum of (y[i] + y[i+1]) * dx / 2
        dx = sweep[1] - sweep[0]
        auc = np.sum((accuracies[:-1] + accuracies[1:]) * dx / 2) / max_thresh
        auc_results[max_thresh] = float(auc)

    return auc_results


def compute_camera_pose_metrics(pred_extrinsics, gt_extrinsics):
    """
    Compute camera pose metrics (RRA, RTA, AUC) over all pairs of views.

    Args:
        pred_extrinsics: (N, 3, 4) predicted camera extrinsics (world-to-camera)
        gt_extrinsics: (N, 3, 4) GT camera extrinsics (world-to-camera)

    Returns:
        Dictionary with RRA, RTA, AUC metrics
    """
    N = len(pred_extrinsics)
    if N < 2:
        return None

    R_pred = pred_extrinsics[:, :3, :3]
    t_pred = pred_extrinsics[:, :3, 3]
    R_gt = gt_extrinsics[:, :3, :3]
    t_gt = gt_extrinsics[:, :3, 3]

    rra_errors = []
    rta_errors = []
    pair_errors = []

    for i, j in combinations(range(N), 2):
        rra = compute_relative_rotation_error(R_pred[i], R_pred[j], R_gt[i], R_gt[j])
        rta = compute_relative_translation_error(t_pred[i], t_pred[j], t_gt[i], t_gt[j])
        pair_error = max(rra, rta)

        rra_errors.append(rra)
        rta_errors.append(rta)
        pair_errors.append(pair_error)

    auc_results = compute_auc(pair_errors, [3, 5, 10, 30])

    return {
        "num_pairs": int(len(pair_errors)),
        "rra_mean": round(float(np.mean(rra_errors)), 4),
        "rta_mean": round(float(np.mean(rta_errors)), 4),
        "pose_error_mean": round(float(np.mean(pair_errors)), 4),
        "rra_median": round(float(np.median(rra_errors)), 4),
        "rta_median": round(float(np.median(rta_errors)), 4),
        "pose_error_median": round(float(np.median(pair_errors)), 4),
        "auc_3": round(float(auc_results[3]), 4),
        "auc_5": round(float(auc_results[5]), 4),
        "auc_10": round(float(auc_results[10]), 4),
        "auc_30": round(float(auc_results[30]), 4),
    }


def process_run_directory(run_dir, dry_run=False):
    """
    Process a single run directory: read cameras.json, compute metrics, update metrics.json.
    Returns (success, message).
    """
    cameras_path = run_dir / "cameras.json"
    metrics_path = run_dir / "metrics.json"

    if not cameras_path.exists():
        return False, "No cameras.json"

    if not metrics_path.exists():
        return False, "No metrics.json"

    # Load cameras
    with open(cameras_path) as f:
        cameras_data = json.load(f)

    pred_cameras = cameras_data.get("pred_cameras", [])
    gt_cameras = cameras_data.get("gt_cameras", [])

    if not gt_cameras:
        return False, "No GT cameras"

    if len(pred_cameras) != len(gt_cameras):
        return False, f"Mismatch: {len(pred_cameras)} pred vs {len(gt_cameras)} gt"

    if len(pred_cameras) < 2:
        return False, "Need at least 2 cameras"

    # Extract extrinsics (3x4 matrices)
    try:
        pred_extrinsics = np.array([c["extrinsic"] for c in pred_cameras])
        gt_extrinsics = np.array([c["extrinsic"] for c in gt_cameras])
    except (KeyError, ValueError) as e:
        return False, f"Error extracting extrinsics: {e}"

    # Compute camera pose metrics
    camera_pose_metrics = compute_camera_pose_metrics(pred_extrinsics, gt_extrinsics)

    if camera_pose_metrics is None:
        return False, "Failed to compute metrics"

    if dry_run:
        return True, f"Would add: RRA={camera_pose_metrics['rra_mean']:.2f}°, RTA={camera_pose_metrics['rta_mean']:.2f}°, AUC@10={camera_pose_metrics['auc_10']:.4f}, AUC@30={camera_pose_metrics['auc_30']:.4f}"

    # Load existing metrics and update
    with open(metrics_path) as f:
        metrics = json.load(f)

    metrics["camera_pose"] = camera_pose_metrics

    # Write updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return True, f"RRA={camera_pose_metrics['rra_mean']:.2f}°, RTA={camera_pose_metrics['rta_mean']:.2f}°, AUC@10={camera_pose_metrics['auc_10']:.4f}, AUC@30={camera_pose_metrics['auc_30']:.4f}"


def find_run_directories(results_dir):
    """Find all run directories (those containing cameras.json)."""
    run_dirs = []
    for root, dirs, files in os.walk(results_dir):
        if "cameras.json" in files:
            run_dirs.append(Path(root))
    return sorted(run_dirs)


def main():
    parser = argparse.ArgumentParser(description="Recompute camera pose metrics for existing data")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    run_dirs = find_run_directories(results_dir)
    print(f"Found {len(run_dirs)} run directories")

    if args.dry_run:
        print("\n=== DRY RUN (no files will be modified) ===\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for run_dir in run_dirs:
        rel_path = run_dir.relative_to(results_dir)
        success, message = process_run_directory(run_dir, dry_run=args.dry_run)

        if success:
            print(f"✓ {rel_path}: {message}")
            success_count += 1
        elif "No GT" in message or "Mismatch" in message:
            print(f"⊘ {rel_path}: {message}")
            skip_count += 1
        else:
            print(f"✗ {rel_path}: {message}")
            error_count += 1

    print(f"\n{'='*60}")
    print(f"Summary: {success_count} updated, {skip_count} skipped, {error_count} errors")
    if args.dry_run:
        print("(dry run - no files were modified)")


if __name__ == "__main__":
    main()
