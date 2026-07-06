#!/usr/bin/env python3
"""
Export VGGT inference results to static web viewer format.
Saves raw data (depth maps, point maps, confidence) for on-demand projection in viewer.

Usage:
    python export_to_viewer.py --image_folder /path/to/data --output results --tags vggt
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms as TF
from itertools import combinations

# Try to import scipy's cKDTree, fall back to pure numpy implementation
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except (ImportError, ValueError):
    HAS_SCIPY = False

sys.path.insert(0, "/home/chuanruo/vggt_train")
sys.path.insert(0, "/home/chuanruo/vggt_train/training")

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from data.dataset_util import resize_image_depth_and_intrinsic, crop_image_depth_and_intrinsic_by_pp

# =============================================================================
# DEFAULTS (same as eval_viser.py)
# =============================================================================
CHECKPOINT = "/home/chuanruo/vggt_train/vggt_checkpoints/model.pt"
IMG_SIZE = 518
USE_LORA = False
FRAME_INDICES = [1, 3, 5, 7, 9, 17, 23]
# =============================================================================


# =============================================================================
# CAMERA POSE METRICS (RRA, RTA, AUC)
# =============================================================================

def rotation_matrix_to_angle(R):
    """Convert rotation matrix to angle in degrees using Rodrigues formula."""
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle_rad)


def compute_relative_rotation_error(R_pred_i, R_pred_j, R_gt_i, R_gt_j):
    """
    Compute RRA (Relative Rotation Accuracy) between a pair of views.

    RRA is the geodesic angle between:
      R_rel_pred = R_j @ R_i.T  (predicted relative rotation)
      R_rel_gt = R_j_gt @ R_i_gt.T  (GT relative rotation)

    Returns error in degrees.
    """
    R_rel_pred = R_pred_j @ R_pred_i.T
    R_rel_gt = R_gt_j @ R_gt_i.T
    # Error rotation: R_rel_pred @ R_rel_gt.T
    R_error = R_rel_pred @ R_rel_gt.T
    return rotation_matrix_to_angle(R_error)


def compute_relative_translation_error(t_pred_i, t_pred_j, t_gt_i, t_gt_j):
    """
    Compute RTA (Relative Translation Accuracy) between a pair of views.

    RTA is the angle between relative translation directions (unit vectors).
    Scale is ambiguous, so we only compare directions.

    Returns error in degrees.
    """
    # Relative translation vectors
    t_rel_pred = t_pred_j - t_pred_i
    t_rel_gt = t_gt_j - t_gt_i

    # Normalize to unit vectors
    norm_pred = np.linalg.norm(t_rel_pred)
    norm_gt = np.linalg.norm(t_rel_gt)

    if norm_pred < 1e-8 or norm_gt < 1e-8:
        return 0.0  # Both cameras at same position, no meaningful error

    t_rel_pred = t_rel_pred / norm_pred
    t_rel_gt = t_rel_gt / norm_gt

    # Angle between unit vectors
    cos_angle = np.clip(np.dot(t_rel_pred, t_rel_gt), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def compute_auc(errors, thresholds):
    """
    Compute Area Under the Curve for accuracy-vs-threshold.

    Args:
        errors: List of error values (in degrees)
        thresholds: List of threshold values to compute AUC at (e.g., [3, 5, 10, 30])

    Returns:
        Dictionary mapping threshold to AUC value (normalized to [0, 1])
    """
    if len(errors) == 0:
        return {t: 0.0 for t in thresholds}

    errors = np.array(errors)
    auc_results = {}

    for max_thresh in thresholds:
        # Sweep threshold from 0 to max_thresh
        # Use fine granularity for accurate integration
        sweep = np.linspace(0, max_thresh, 100)
        accuracies = []
        for tau in sweep:
            acc = np.mean(errors < tau)
            accuracies.append(acc)

        # Integrate using trapezoidal rule, normalized by threshold range
        auc = np.trapz(accuracies, sweep) / max_thresh
        auc_results[max_thresh] = float(auc)

    return auc_results


def compute_camera_pose_metrics(pred_extrinsics, gt_extrinsics):
    """
    Compute camera pose metrics (RRA, RTA, AUC) over all pairs of views.

    Args:
        pred_extrinsics: (N, 3, 4) predicted camera extrinsics (world-to-camera)
        gt_extrinsics: (N, 3, 4) GT camera extrinsics (world-to-camera)

    Returns:
        Dictionary with:
        - rra_mean: Mean RRA over all pairs
        - rta_mean: Mean RTA over all pairs
        - pose_error_mean: Mean of max(RRA, RTA) per pair
        - auc_3, auc_5, auc_10, auc_30: AUC at different thresholds
        - per_pair: List of per-pair errors
    """
    N = len(pred_extrinsics)
    if N < 2:
        return None

    # Extract R and t from extrinsics (world-to-camera: [R | t])
    R_pred = pred_extrinsics[:, :3, :3]  # N x 3 x 3
    t_pred = pred_extrinsics[:, :3, 3]   # N x 3
    R_gt = gt_extrinsics[:, :3, :3]
    t_gt = gt_extrinsics[:, :3, 3]

    rra_errors = []
    rta_errors = []
    pair_errors = []  # max(RRA, RTA) per pair
    per_pair_details = []

    # All N-choose-2 pairs
    for i, j in combinations(range(N), 2):
        rra = compute_relative_rotation_error(R_pred[i], R_pred[j], R_gt[i], R_gt[j])
        rta = compute_relative_translation_error(t_pred[i], t_pred[j], t_gt[i], t_gt[j])
        pair_error = max(rra, rta)

        rra_errors.append(rra)
        rta_errors.append(rta)
        pair_errors.append(pair_error)
        per_pair_details.append({
            "pair": (i, j),
            "rra": round(rra, 4),
            "rta": round(rta, 4),
            "max_error": round(pair_error, 4)
        })

    # Compute AUC at standard thresholds
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


# =============================================================================
# POINT CLOUD METRICS (Chamfer Distance with Umeyama Alignment)
# =============================================================================

def umeyama_alignment(src_points, dst_points, with_scale=True):
    """
    Umeyama algorithm for computing optimal similarity transform (rotation + translation + scale).

    Aligns src_points to dst_points by finding optimal R, t, s such that:
        dst ≈ s * R @ src + t

    Args:
        src_points: (N, 3) source points to be transformed
        dst_points: (N, 3) target points
        with_scale: If True, estimate scale. If False, scale=1.

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scalar scale factor
    """
    assert src_points.shape == dst_points.shape
    n, dim = src_points.shape

    # Compute centroids
    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)

    # Center the points
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    # Compute variances
    src_var = np.sum(src_centered ** 2) / n

    # Compute covariance matrix
    cov = (dst_centered.T @ src_centered) / n

    # SVD
    U, S, Vt = np.linalg.svd(cov)

    # Handle reflection case
    d = np.ones(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1

    # Rotation
    R = U @ np.diag(d) @ Vt

    # Scale
    if with_scale:
        s = np.sum(S * d) / src_var
    else:
        s = 1.0

    # Translation
    t = dst_mean - s * (R @ src_mean)

    return R, t, s


def apply_transform(points, R, t, s):
    """Apply similarity transform: s * R @ p + t"""
    return s * (points @ R.T) + t


def nearest_neighbor_distance_numpy(query_points, target_points, batch_size=1000):
    """
    Pure NumPy implementation of nearest neighbor distance computation.
    Uses batched computation to avoid memory issues with large point clouds.

    Args:
        query_points: (N, 3) points to query from
        target_points: (M, 3) points to find nearest neighbors in

    Returns:
        (N,) array of distances to nearest neighbor in target_points
    """
    n_query = len(query_points)
    distances = np.zeros(n_query)

    for i in range(0, n_query, batch_size):
        end = min(i + batch_size, n_query)
        batch = query_points[i:end]  # (batch, 3)
        # Compute all pairwise distances: (batch, M)
        diff = batch[:, np.newaxis, :] - target_points[np.newaxis, :, :]  # (batch, M, 3)
        dists = np.linalg.norm(diff, axis=2)  # (batch, M)
        distances[i:end] = np.min(dists, axis=1)

    return distances


def compute_chamfer_distance(pred_points, gt_points, pred_mask=None, gt_mask=None, align=True):
    """
    Compute Chamfer distance metrics between predicted and GT point clouds.

    Args:
        pred_points: (H, W, 3) or (N, 3) predicted point cloud
        gt_points: (H, W, 3) or (M, 3) GT point cloud
        pred_mask: Optional mask for valid predicted points
        gt_mask: Optional mask for valid GT points
        align: If True, align pred to GT using Umeyama algorithm first

    Returns:
        Dictionary with:
        - accuracy: Mean distance from pred to nearest GT (pred completeness)
        - completeness: Mean distance from GT to nearest pred (coverage)
        - overall: (accuracy + completeness) / 2
        - scale: Estimated scale factor (if aligned)
    """
    # Flatten to (N, 3)
    pred_flat = pred_points.reshape(-1, 3)
    gt_flat = gt_points.reshape(-1, 3)

    # Apply masks
    if pred_mask is not None:
        pred_mask_flat = pred_mask.reshape(-1).astype(bool)
        pred_flat = pred_flat[pred_mask_flat]
    if gt_mask is not None:
        gt_mask_flat = gt_mask.reshape(-1).astype(bool)
        gt_flat = gt_flat[gt_mask_flat]

    # Filter invalid points (NaN, Inf, zeros)
    pred_valid = np.isfinite(pred_flat).all(axis=1) & (np.abs(pred_flat).sum(axis=1) > 1e-8)
    gt_valid = np.isfinite(gt_flat).all(axis=1) & (np.abs(gt_flat).sum(axis=1) > 1e-8)
    pred_flat = pred_flat[pred_valid]
    gt_flat = gt_flat[gt_valid]

    if len(pred_flat) < 10 or len(gt_flat) < 10:
        return None

    # Subsample if too many points (for efficiency)
    max_points = 50000
    if len(pred_flat) > max_points:
        indices = np.random.choice(len(pred_flat), max_points, replace=False)
        pred_flat = pred_flat[indices]
    if len(gt_flat) > max_points:
        indices = np.random.choice(len(gt_flat), max_points, replace=False)
        gt_flat = gt_flat[indices]

    scale = 1.0

    # Align pred to GT using Umeyama
    if align and len(pred_flat) >= 3 and len(gt_flat) >= 3:
        # Use a subset of corresponding points for alignment
        # Since we don't have correspondences, use centroids and PCA-based alignment
        # or just use the full point clouds with Umeyama on random subsets

        # Simple approach: subsample both to same size, find optimal transform
        n_align = min(len(pred_flat), len(gt_flat), 5000)
        pred_sample = pred_flat[np.random.choice(len(pred_flat), n_align, replace=False)]
        gt_sample = gt_flat[np.random.choice(len(gt_flat), n_align, replace=False)]

        # Use ICP-like alignment: find nearest neighbors, then Umeyama
        # For simplicity, use centroid alignment + scale estimation
        pred_centroid = np.mean(pred_flat, axis=0)
        gt_centroid = np.mean(gt_flat, axis=0)

        pred_centered = pred_flat - pred_centroid
        gt_centered = gt_flat - gt_centroid

        # Estimate scale from spread
        pred_scale = np.sqrt(np.mean(np.sum(pred_centered ** 2, axis=1)))
        gt_scale = np.sqrt(np.mean(np.sum(gt_centered ** 2, axis=1)))

        if pred_scale > 1e-8:
            scale = gt_scale / pred_scale

        # Apply centering and scaling
        pred_aligned = pred_centered * scale + gt_centroid
    else:
        pred_aligned = pred_flat

    # Compute nearest neighbor distances
    if HAS_SCIPY:
        # Use KD-trees for efficient nearest neighbor lookup
        gt_tree = cKDTree(gt_flat)
        pred_tree = cKDTree(pred_aligned)
        dist_pred_to_gt, _ = gt_tree.query(pred_aligned, k=1)
        dist_gt_to_pred, _ = pred_tree.query(gt_flat, k=1)
    else:
        # Fallback to pure numpy (slower but no dependencies)
        dist_pred_to_gt = nearest_neighbor_distance_numpy(pred_aligned, gt_flat)
        dist_gt_to_pred = nearest_neighbor_distance_numpy(gt_flat, pred_aligned)

    # Accuracy: mean distance from pred to nearest GT
    accuracy = float(np.mean(dist_pred_to_gt))

    # Completeness: mean distance from GT to nearest pred
    completeness = float(np.mean(dist_gt_to_pred))

    # Overall (symmetric)
    overall = (accuracy + completeness) / 2

    return {
        "accuracy": round(float(accuracy), 6),
        "completeness": round(float(completeness), 6),
        "overall": round(float(overall), 6),
        "scale": round(float(scale), 4),
        "num_pred_points": int(len(pred_aligned)),
        "num_gt_points": int(len(gt_flat)),
    }


def compute_chamfer_metrics_all_frames(pred_world_points, gt_world_points, gt_point_masks=None):
    """
    Compute Chamfer metrics aggregated across all frames.

    Args:
        pred_world_points: (S, H, W, 3) predicted point maps
        gt_world_points: (S, H, W, 3) GT point maps
        gt_point_masks: Optional (S, H, W) validity masks

    Returns:
        Dictionary with chamfer metrics
    """
    S = len(pred_world_points)

    # Aggregate all points across frames
    all_pred = []
    all_gt = []

    for i in range(S):
        pred_pts = pred_world_points[i].reshape(-1, 3)
        gt_pts = gt_world_points[i].reshape(-1, 3)

        if gt_point_masks is not None and i < len(gt_point_masks):
            mask = gt_point_masks[i].reshape(-1).astype(bool)
            pred_pts = pred_pts[mask]
            gt_pts = gt_pts[mask]

        all_pred.append(pred_pts)
        all_gt.append(gt_pts)

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    # Compute on aggregated point cloud
    result = compute_chamfer_distance(all_pred, all_gt, align=True)

    if result is None:
        return None

    return {
        "chamfer_accuracy": float(result["accuracy"]),
        "chamfer_completeness": float(result["completeness"]),
        "chamfer_overall": float(result["overall"]),
        "alignment_scale": float(result["scale"]),
        "num_pred_points": int(result["num_pred_points"]),
        "num_gt_points": int(result["num_gt_points"]),
    }


def compute_metrics(
    pred_extrinsics,
    pred_depth,
    pred_world_points,
    pred_intrinsics,
    gt_extrinsic=None,
    gt_depths=None,
    gt_world_points=None,
    gt_point_masks=None,
    frame_labels=None,
    H=None, W=None,
    pred_pose_enc=None,
    gt_pose_enc=None,
):
    """
    Compute pose, depth, and pointmap metrics comparing predictions to ground truth.
    Pose metrics use pose encoding (same as eval_viser.py print_errors).
    Returns a dict suitable for metrics.json.
    """
    S = len(pred_extrinsics)
    labels = frame_labels if frame_labels else [f"t{i}" for i in range(S)]

    metrics = {
        "num_frames": S,
        "conf_threshold": 5.0,
    }

    # === Pose metrics (using pose encoding, same as eval_viser.py print_errors) ===
    if gt_pose_enc is not None and pred_pose_enc is not None:
        pose_per_frame = []

        # Convert to numpy if needed
        gt = gt_pose_enc[0].cpu().numpy() if hasattr(gt_pose_enc, 'cpu') else gt_pose_enc[0] if len(gt_pose_enc.shape) > 2 else gt_pose_enc
        pred = pred_pose_enc[0] if len(pred_pose_enc.shape) > 2 else pred_pose_enc
        if hasattr(pred, 'cpu'):
            pred = pred.cpu().numpy()

        for i in range(min(S, len(gt))):
            # Same as eval_viser.py print_errors:
            # loss_T = mean absolute error of translation encoding (first 3 values)
            # loss_R = mean absolute error of rotation encoding (values 3:7)
            # loss_FL = mean absolute error of focal length encoding (values 7:)
            loss_T = float(np.abs(gt[i, :3] - pred[i, :3]).mean())
            loss_R = float(np.abs(gt[i, 3:7] - pred[i, 3:7]).mean())
            loss_FL = float(np.abs(gt[i, 7:] - pred[i, 7:]).mean())

            pose_per_frame.append({
                "frame": labels[i],
                "loss_T": round(loss_T, 6),
                "loss_R": round(loss_R, 6),
                "loss_FL": round(loss_FL, 6),
            })

        metrics["pose"] = {
            "per_frame": pose_per_frame,
            "mean_loss_T": round(np.mean([p["loss_T"] for p in pose_per_frame]), 6),
            "mean_loss_R": round(np.mean([p["loss_R"] for p in pose_per_frame]), 6),
            "mean_loss_FL": round(np.mean([p["loss_FL"] for p in pose_per_frame]), 6),
        }

    # === Depth metrics ===
    if gt_depths is not None and len(gt_depths) > 0:
        depth_per_frame = []
        gt_depths_sq = gt_depths.squeeze(-1) if gt_depths.ndim == 4 else gt_depths
        pred_depth_sq = pred_depth.squeeze(-1) if pred_depth.ndim == 4 else pred_depth

        for i in range(min(S, len(gt_depths_sq))):
            pred_d = pred_depth_sq[i]
            gt_d = gt_depths_sq[i]

            # Use mask if available
            if gt_point_masks is not None and i < len(gt_point_masks):
                mask = gt_point_masks[i].astype(bool)
            else:
                mask = (gt_d > 0) & (pred_d > 0)

            if mask.sum() > 0:
                diff = np.abs(pred_d[mask] - gt_d[mask])
                mae = float(np.mean(diff))
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                n_pixels = int(mask.sum())
            else:
                mae, rmse, n_pixels = 0.0, 0.0, 0

            depth_per_frame.append({
                "frame": labels[i],
                "depth_mae": round(mae, 6),
                "depth_rmse": round(rmse, 6),
                "n_pixels": n_pixels,
            })

        metrics["depth"] = {
            "per_frame": depth_per_frame,
            "mean_depth_mae": round(np.mean([d["depth_mae"] for d in depth_per_frame]), 6),
            "mean_depth_rmse": round(np.mean([d["depth_rmse"] for d in depth_per_frame]), 6),
        }

    # === Pointmap metrics ===
    if gt_world_points is not None and len(gt_world_points) > 0:
        pointmap_per_frame = []

        for i in range(min(S, len(gt_world_points))):
            pred_pts = pred_world_points[i]  # H x W x 3
            gt_pts = gt_world_points[i]  # H x W x 3

            # Use mask if available
            if gt_point_masks is not None and i < len(gt_point_masks):
                mask = gt_point_masks[i].astype(bool)
            else:
                mask = np.ones(pred_pts.shape[:2], dtype=bool)

            if mask.sum() > 0:
                diff = np.linalg.norm(pred_pts[mask] - gt_pts[mask], axis=-1)
                mae = float(np.mean(diff))
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                n_points = int(mask.sum())
            else:
                mae, rmse, n_points = 0.0, 0.0, 0

            pointmap_per_frame.append({
                "frame": labels[i],
                "pointmap_mae": round(mae, 6),
                "pointmap_rmse": round(rmse, 6),
                "n_points": n_points,
            })

        metrics["pointmap"] = {
            "per_frame": pointmap_per_frame,
            "mean_pointmap_mae": round(np.mean([p["pointmap_mae"] for p in pointmap_per_frame]), 6),
            "mean_pointmap_rmse": round(np.mean([p["pointmap_rmse"] for p in pointmap_per_frame]), 6),
        }

    # === Camera pose metrics (RRA, RTA, AUC) ===
    if gt_extrinsic is not None and len(gt_extrinsic) >= 2:
        camera_pose_metrics = compute_camera_pose_metrics(pred_extrinsics, gt_extrinsic)
        if camera_pose_metrics is not None:
            metrics["camera_pose"] = camera_pose_metrics

    # === Chamfer distance metrics (Accuracy, Completeness, Overall) ===
    if gt_world_points is not None and len(gt_world_points) > 0:
        chamfer_metrics = compute_chamfer_metrics_all_frames(
            pred_world_points, gt_world_points, gt_point_masks
        )
        if chamfer_metrics is not None:
            metrics["chamfer"] = chamfer_metrics

    return metrics


def load_images_training_style(image_paths, img_size=IMG_SIZE, safe_bound=4):
    """Preprocess flat folder images the same way as training."""
    images = []
    to_tensor = TF.ToTensor()
    for image_path in sorted(image_paths):
        img = Image.open(image_path)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        img = np.array(img)
        h, w = img.shape[:2]
        original_size = np.array([h, w])
        intrinsic = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
        target_shape = np.array([img_size, img_size])
        img, _, intrinsic, _ = resize_image_depth_and_intrinsic(
            img, np.ones((h, w), dtype=np.float32), intrinsic, target_shape, original_size,
            track=None, safe_bound=safe_bound, rescale_aug=False
        )
        img, _, _, _ = crop_image_depth_and_intrinsic_by_pp(
            img, np.ones(img.shape[:2], dtype=np.float32), intrinsic, target_shape,
            track=None, filepath=image_path, strict=True
        )
        images.append(to_tensor(img))
    return torch.stack(images)


def load_co3d_batch(co3d_dir, annotations_dir, split="train", sequence=None, frame_indices=None, img_size=IMG_SIZE):
    """Load frames from a CO3D sequence via the training pipeline."""
    from data.composed_dataset import ComposedDataset
    from omegaconf import OmegaConf

    dataset_cfg = OmegaConf.create({
        "_target_": "data.datasets.co3d.Co3dDataset",
        "split": split,
        "CO3D_DIR": co3d_dir,
        "CO3D_ANNOTATION_DIR": annotations_dir,
    })

    def make_common(n):
        return OmegaConf.create({
            "img_size": img_size, "patch_size": 14, "debug": False, "repeat_batch": False,
            "fix_img_num": -1, "fix_aspect_ratio": 1.0, "load_track": False, "track_num": 1024,
            "training": False, "inside_random": False, "rescale": True, "rescale_aug": False,
            "landscape_check": False, "get_nearby": True, "load_depth": True,
            "img_nums": [n, n], "max_img_per_gpu": n, "allow_duplicate_img": False,
            "augs": {"cojitter": False, "cojitter_ratio": 0.0, "scales": None,
                     "aspects": [1.0, 1.0], "color_jitter": None, "gray_scale": False, "gau_blur": False},
        })

    composed_tmp = ComposedDataset(dataset_configs=[dataset_cfg], common_config=make_common(1))
    dataset_tmp = composed_tmp.base_dataset.datasets[0]
    available_seqs = dataset_tmp.sequence_list

    if sequence is not None:
        if sequence not in dataset_tmp.data_store:
            raise ValueError(f"Sequence '{sequence}' not found. Available: {available_seqs}")
        seq = sequence
    else:
        seq = available_seqs[0]
        if len(available_seqs) > 1:
            print(f"Multiple sequences found: {available_seqs}. Using '{seq}'. Pass --sequence to select.")

    n_total = len(dataset_tmp.data_store[seq])
    print(f"Found {n_total} {split} frames in sequence '{seq}'")

    if frame_indices is not None:
        all_frame_data = dataset_tmp.data_store[seq]
        subset_frame_data = [all_frame_data[i] for i in frame_indices]
        n_load = len(frame_indices)
        print(f"Loading {n_load} frames at indices {frame_indices}")
    else:
        n_load = n_total

    composed_all = ComposedDataset(dataset_configs=[dataset_cfg], common_config=make_common(n_load))
    dataset_all = composed_all.base_dataset.datasets[0]
    for s in list(dataset_all.data_store.keys()):
        if s != seq:
            del dataset_all.data_store[s]
    dataset_all.sequence_list = [seq]

    if frame_indices is not None:
        dataset_all.data_store[seq] = subset_frame_data
    else:
        dataset_all.data_store[seq] = dataset_all.data_store[seq][:n_total]

    sample = composed_all[(0, n_load, 1.0)]
    result = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
    result["_frame_indices"] = frame_indices
    return result


def get_gt_cameras(batch_all, frame_indices=None):
    """Normalize GT using only the subset frames."""
    if frame_indices is not None:
        batch = {k: v[:, frame_indices] if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == batch_all["images"].shape[1] else v
                 for k, v in batch_all.items()}
    else:
        batch = batch_all

    extrinsics = batch["extrinsics"]
    intrinsics = batch["intrinsics"]
    world_points = batch["world_points"]
    cam_points = batch.get("cam_points")
    depths = batch.get("depths")
    point_masks = batch.get("point_masks")

    norm_ext, _, norm_world_points, norm_depths = normalize_camera_extrinsics_and_points_batch(
        extrinsics=extrinsics.clone(),
        cam_points=cam_points,
        world_points=world_points.clone(),
        depths=depths,
        point_masks=point_masks,
    )
    B, S, C, H, W = batch["images"].shape
    gt_pose_enc = extri_intri_to_pose_encoding(norm_ext, intrinsics, image_size_hw=(H, W))
    gt_extrinsic, _ = pose_encoding_to_extri_intri(gt_pose_enc, (H, W))
    norm_depths_np = norm_depths[0].cpu().numpy() if norm_depths is not None else None
    point_masks_np = point_masks[0].bool().cpu().numpy() if point_masks is not None else None
    norm_world_points_np = norm_world_points[0].cpu().numpy() if norm_world_points is not None else None
    return gt_pose_enc, gt_extrinsic[0].float().cpu().numpy(), norm_depths_np, point_masks_np, norm_world_points_np


def export_raw_data(
    pred_dict,
    output_dir: Path,
    gt_extrinsic=None,
    gt_world_points=None,
    gt_depths=None,
    frame_labels=None,
    gt_point_masks=None,
    original_images=None,
    gt_pose_enc=None,
):
    """
    Export raw inference data for on-demand projection in viewer.

    Saves:
    - images/*.png (RGB)
    - depths/*.bin (predicted depth per frame, float32)
    - pointmaps/*.bin (world_points per frame, float32 HxWx3)
    - confidence/*.bin (confidence per frame, float32)
    - cameras.json (extrinsics, intrinsics, metadata)
    - gt_depths/*.bin (GT depth if available)
    - gt_pointmaps/*.bin (GT world_points if available)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract predictions
    images = pred_dict["images"]  # S x C x H x W
    world_points = pred_dict["world_points"]  # S x H x W x 3
    world_points_conf = pred_dict["world_points_conf"]  # S x H x W
    depth = pred_dict["depth"]  # S x H x W x 1
    depth_conf = pred_dict["depth_conf"]  # S x H x W
    extrinsics = pred_dict["extrinsic"]  # S x 3 x 4
    intrinsics = pred_dict["intrinsic"]  # S x 3 x 3

    S, C, H, W = images.shape
    labels = frame_labels if frame_labels is not None else [f"t{i}" for i in range(S)]

    # Compute scene center from all world points (for centering in viewer)
    all_points = world_points.reshape(-1, 3)
    scene_center = np.mean(all_points, axis=0)

    # Camera transforms (centered)
    cam_to_world_mat = closed_form_inverse_se3(extrinsics)
    cam_to_world = cam_to_world_mat[:, :3, :].copy()
    cam_to_world[..., -1] -= scene_center

    print(f"  Scene center: {scene_center}")
    print(f"  Depth range: {depth.min():.3f} - {depth.max():.3f}")
    print(f"  Confidence range: {world_points_conf.min():.3f} - {world_points_conf.max():.3f}")

    # Create directories
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "depths").mkdir(exist_ok=True)
    (output_dir / "pointmaps").mkdir(exist_ok=True)
    (output_dir / "confidence").mkdir(exist_ok=True)
    (output_dir / "depth_conf").mkdir(exist_ok=True)

    # Save per-frame data
    for i in range(S):
        # RGB image
        img = images[i].transpose(1, 2, 0)  # H x W x C
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(output_dir / "images" / f"view_{i:03d}.png")

        # Depth (float32 binary)
        depth_i = depth[i].squeeze(-1).astype(np.float32)  # H x W
        depth_i.tofile(output_dir / "depths" / f"view_{i:03d}.bin")

        # Point map (float32 binary, H x W x 3)
        pointmap_i = (world_points[i] - scene_center).astype(np.float32)  # Center points
        pointmap_i.tofile(output_dir / "pointmaps" / f"view_{i:03d}.bin")

        # Confidence for world_points
        conf_i = world_points_conf[i].astype(np.float32)  # H x W
        conf_i.tofile(output_dir / "confidence" / f"view_{i:03d}.bin")

        # Confidence for depth
        depth_conf_i = depth_conf[i].astype(np.float32)  # H x W
        depth_conf_i.tofile(output_dir / "depth_conf" / f"view_{i:03d}.bin")

    print(f"  Wrote {S} frames (images, depths, pointmaps, confidence)")

    # Save GT data if available
    if gt_world_points is not None:
        (output_dir / "gt_pointmaps").mkdir(exist_ok=True)
        for i in range(min(S, len(gt_world_points))):
            gt_pm_i = (gt_world_points[i] - scene_center).astype(np.float32)
            gt_pm_i.tofile(output_dir / "gt_pointmaps" / f"view_{i:03d}.bin")
        print(f"  Wrote {min(S, len(gt_world_points))} GT pointmaps")

    if gt_depths is not None:
        (output_dir / "gt_depths").mkdir(exist_ok=True)
        gt_depths_sq = gt_depths.squeeze(-1) if gt_depths.ndim == 4 else gt_depths
        for i in range(min(S, len(gt_depths_sq))):
            gt_d_i = gt_depths_sq[i].astype(np.float32)
            gt_d_i.tofile(output_dir / "gt_depths" / f"view_{i:03d}.bin")
        print(f"  Wrote {min(S, len(gt_depths_sq))} GT depths")

    if gt_point_masks is not None:
        (output_dir / "gt_masks").mkdir(exist_ok=True)
        for i in range(min(S, len(gt_point_masks))):
            mask_i = gt_point_masks[i].astype(np.uint8)
            mask_i.tofile(output_dir / "gt_masks" / f"view_{i:03d}.bin")
        print(f"  Wrote {min(S, len(gt_point_masks))} GT masks")

    # Build cameras data
    pred_cameras = []
    for i in range(S):
        # FOV calculation (same as viser)
        fy = 1.1 * H
        fov_rad = 2 * np.arctan2(H / 2, fy)
        fov_deg = float(np.degrees(fov_rad))

        pred_cameras.append({
            "view_id": labels[i],
            "position": cam_to_world[i, :, 3].tolist(),
            "matrix": cam_to_world[i].tolist(),  # 3x4 cam-to-world (centered)
            "extrinsic": extrinsics[i].tolist(),  # 3x4 world-to-cam (original)
            "intrinsic": intrinsics[i].tolist(),  # 3x3
            "fov_deg": round(fov_deg, 1),
            "aspect": round(W / H, 3),
            "source_image": original_images[i] if original_images and i < len(original_images) else None
        })

    # GT cameras if available
    gt_cameras = []
    if gt_extrinsic is not None:
        gt_cam2world = closed_form_inverse_se3(gt_extrinsic)[:, :3, :]
        gt_cam2world[..., -1] -= scene_center

        for i in range(min(S, len(gt_extrinsic))):
            fov_rad = 2 * np.arctan2(H / 2, 1.1 * H)
            fov_deg = float(np.degrees(fov_rad))

            gt_cameras.append({
                "view_id": labels[i],
                "position": gt_cam2world[i, :, 3].tolist(),
                "matrix": gt_cam2world[i].tolist(),
                "extrinsic": gt_extrinsic[i].tolist(),
                "intrinsic": intrinsics[i].tolist(),
                "fov_deg": round(fov_deg, 1),
                "aspect": round(W / H, 3),
            })

    # Write cameras.json with all metadata
    cameras_data = {
        "num_frames": S,
        "height": H,
        "width": W,
        "scene_center": scene_center.tolist(),
        "depth_range": [float(depth.min()), float(depth.max())],
        "conf_range": [float(world_points_conf.min()), float(world_points_conf.max())],
        "pred_cameras": pred_cameras,
        "gt_cameras": gt_cameras if gt_cameras else None,
        "has_gt_pointmaps": gt_world_points is not None,
        "has_gt_depths": gt_depths is not None,
        "has_gt_masks": gt_point_masks is not None,
    }

    with open(output_dir / "cameras.json", 'w') as f:
        json.dump(cameras_data, f, indent=2)
    print(f"  Wrote cameras.json")

    # === Compute and write metrics.json ===
    # pred_pose_enc is in pred_dict (as numpy array after squeeze)
    pred_pose_enc = pred_dict.get("pose_enc")
    metrics = compute_metrics(
        pred_extrinsics=extrinsics,
        pred_depth=depth,
        pred_world_points=world_points,
        pred_intrinsics=intrinsics,
        gt_extrinsic=gt_extrinsic,
        gt_depths=gt_depths,
        gt_world_points=gt_world_points,
        gt_point_masks=gt_point_masks,
        frame_labels=labels,
        H=H, W=W,
        pred_pose_enc=pred_pose_enc,
        gt_pose_enc=gt_pose_enc,
    )
    # Add additional info
    metrics["conf_min"] = float(world_points_conf.min())
    metrics["conf_max"] = float(world_points_conf.max())
    metrics["mean_depth"] = float(depth.mean())
    metrics["split"] = "train"  # Default, could be passed as param

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Wrote metrics.json")


def get_object_id(checkpoint_path: str, data_path: str) -> tuple:
    """Extract object_id from checkpoint + data paths. Returns (object_id, checkpoint_id, data_id)."""
    # Get checkpoint_id - extract just exp name and checkpoint number
    ckpt = Path(checkpoint_path).resolve()
    ckpt_name = ckpt.stem  # e.g., "checkpoint_190" (removes .pt)
    # Look for exp name in parent directories
    exp_name = None
    for part in ckpt.parts:
        if part.startswith("exp"):
            exp_name = part
            break
    if exp_name:
        checkpoint_id = f"{exp_name}_ckpts_{ckpt_name}"  # e.g., "exp197_ckpts_checkpoint_190"
    else:
        checkpoint_id = ckpt_name

    # Get data_id - extract run name and subset
    data = Path(data_path).resolve()
    data_bases = ["/home/chuanruo/TGGT/out/", "/home/chuanruo/TGGT/out"]
    data_str = str(data)
    for base in data_bases:
        if data_str.startswith(base):
            data_str = data_str[len(base):]
            break
    # Flatten: replace / with _ and skip "subsets" directory
    data_parts = [p for p in data_str.strip("/").split("/") if p != "subsets"]
    data_id = "_".join(data_parts)  # e.g., "baboon_run_20260613_014414_my_data_125"

    # Combined object_id: checkpoint/data (one level of nesting)
    object_id = f"{checkpoint_id}/{data_id}"
    return object_id, checkpoint_id, data_id


def update_index(output_dir: Path, checkpoint_id: str):
    """Add checkpoint ID to index.json (list of checkpoints to scan)."""
    index_path = output_dir / "index.json"
    index = []
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    if checkpoint_id not in index:
        index.append(checkpoint_id)
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


def update_runs(object_dir: Path,
                checkpoint: str = None, data_path: str = None, name: str = None):
    """Update runs.json for an object. Just metadata, no tags."""
    runs_path = object_dir / "runs.json"
    runs = {}
    if runs_path.exists():
        with open(runs_path) as f:
            runs = json.load(f)
    if name:
        runs["name"] = name
    if checkpoint:
        runs["checkpoint"] = checkpoint
    if data_path:
        runs["data_path"] = data_path
    with open(runs_path, 'w') as f:
        json.dump(runs, f, indent=2)


def update_model_tags(output_dir: Path, checkpoint_id: str, object_id: str, run_name: str, tag: str):
    """Update tags.json at checkpoint level. Maps tags to list of run paths relative to checkpoint."""
    tags_path = output_dir / checkpoint_id / "tags.json"
    tags_path.parent.mkdir(parents=True, exist_ok=True)
    tags_data = {"tags": {}}
    if tags_path.exists():
        with open(tags_path) as f:
            tags_data = json.load(f)
    if "tags" not in tags_data:
        tags_data["tags"] = {}

    # Path relative to checkpoint: data_id/run_name
    # object_id is checkpoint_id/data_id, so strip checkpoint_id prefix
    rel_object = object_id[len(checkpoint_id):].strip("/")
    run_path = f"{rel_object}/{run_name}"

    if tag not in tags_data["tags"]:
        tags_data["tags"][tag] = []
    if run_path not in tags_data["tags"][tag]:
        tags_data["tags"][tag].append(run_path)

    with open(tags_path, 'w') as f:
        json.dump(tags_data, f, indent=2)


parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--output", type=str, default="results")
parser.add_argument("--sequence", type=str, default=None)
parser.add_argument("--max_images", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--val", type=int, nargs="*", default=None)
parser.add_argument("--val-only", type=int, nargs="*", default=None)
parser.add_argument("--epoch", type=int, default=0)
parser.add_argument("--tags", type=str, nargs="*", default=None)
parser.add_argument("--img_size", type=int, default=224, help="Image size for inference")
parser.add_argument("--frames", type=int, nargs="*", default=None, help="Frame indices to use (default: 1,3,5,7,9,17,23)")


def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use --frames if provided, else default frame_indices
    frame_indices = args.frames if args.frames else FRAME_INDICES

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (same as eval_viser.py)
    checkpoint_path = args.checkpoint if args.checkpoint else CHECKPOINT
    print(f"Loading model from {checkpoint_path}...")
    model = VGGT()
    if USE_LORA:
        model.apply_lora()
    if checkpoint_path.endswith("vggt_checkpoints/model.pt"):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt)
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        current_state = model.state_dict()
        state = {k: v for k, v in state.items() if k in current_state and v.shape == current_state[k].shape}
        model.load_state_dict(state, strict=False)
    model.eval().to(device)

    # Check for annotations
    annotations_dir = args.image_folder.rstrip("/") + "_annotations"
    has_annotations = os.path.isdir(annotations_dir)

    # Set seed for deterministic data loading (dataset has random shuffling)
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    gt_pose_enc = None
    gt_extrinsic = None
    gt_world_points = None
    gt_depths = None
    frame_labels = None
    gt_point_masks_np = None
    original_images = []

    if has_annotations:
        print("CO3D structure detected — loading via training pipeline...")
        val_indices = args.val if (args.val is not None and len(args.val) > 0) else None
        val_only = args.val_only is not None

        if val_only:
            batch_val = load_co3d_batch(args.image_folder, annotations_dir, split="test", sequence=args.sequence, img_size=args.img_size)
            n_val_total = batch_val["images"].shape[1]
            val_only_indices = args.val_only if len(args.val_only) > 0 else list(range(n_val_total))
            val_images = batch_val["images"][0][val_only_indices]
            frame_labels = [f"v{i}" for i in val_only_indices]
            images = val_images.to(device)
            val_batch_sub = {k: v[:, val_only_indices] if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == batch_val["images"].shape[1] else v
                             for k, v in batch_val.items()}
            gt_pose_enc, gt_extrinsic, gt_depths, gt_point_masks_np, gt_world_points = get_gt_cameras(val_batch_sub, frame_indices=None)
            print(f"Val-only mode: {len(val_only_indices)} val frames {val_only_indices}")
            # Use val indices for run name
            frame_indices = val_only_indices
        else:
            batch_train = load_co3d_batch(args.image_folder, annotations_dir, split="train", sequence=args.sequence, frame_indices=frame_indices, img_size=args.img_size)
            train_images = batch_train["images"][0]
            if frame_indices is not None:
                # Use batch["ids"] to get actual frame order (dataset may shuffle internally)
                ids = batch_train["ids"][0].tolist()
                train_frame_labels = [f"t{frame_indices[i]}" for i in ids]
                print(f"Using {len(frame_indices)} train frames: {frame_indices}")
                print(f"Actual batch order (ids): {ids} -> labels: {train_frame_labels}")
            else:
                train_frame_labels = [f"t{i}" for i in range(train_images.shape[0])]
                print(f"Using all {train_images.shape[0]} train frames")

            if val_indices is not None:
                batch_val = load_co3d_batch(args.image_folder, annotations_dir, split="test", sequence=args.sequence, img_size=args.img_size)
                val_images = batch_val["images"][0][val_indices]
                val_frame_labels = [f"v{i}" for i in val_indices]
                print(f"Using {len(val_indices)} val frames: {val_indices}")
                images = torch.cat([train_images, val_images], dim=0).to(device)
                frame_labels = train_frame_labels + val_frame_labels
                train_idx_for_norm = frame_indices if frame_indices is not None else list(range(batch_train["images"].shape[1]))
                combined_batch = {}
                for k in batch_train.keys():
                    v_train = batch_train[k]
                    v_val = batch_val[k]
                    if isinstance(v_train, torch.Tensor) and v_train.dim() >= 2:
                        combined_batch[k] = torch.cat([v_train[:, train_idx_for_norm], v_val[:, val_indices]], dim=1)
                    else:
                        combined_batch[k] = v_train
                gt_pose_enc, gt_extrinsic, gt_depths, gt_point_masks_np, gt_world_points = get_gt_cameras(combined_batch, frame_indices=None)
            else:
                images = train_images.to(device)
                frame_labels = train_frame_labels
                gt_pose_enc, gt_extrinsic, gt_depths, gt_point_masks_np, gt_world_points = get_gt_cameras(batch_train, frame_indices=None)

        # Find original images
        image_dir = Path(args.image_folder)
        original_images = sorted(glob.glob(str(image_dir / "**/images/*.png"), recursive=True))
        original_images += sorted(glob.glob(str(image_dir / "**/images/*.jpg"), recursive=True))
        if frame_indices and original_images:
            original_images = [original_images[i] for i in frame_indices if i < len(original_images)]

    else:
        print("Flat folder — loading via training-style preprocessing...")
        image_paths = sorted(glob.glob(os.path.join(args.image_folder, "*")))
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if args.max_images is not None:
            image_paths = image_paths[:args.max_images]
        if not image_paths:
            raise RuntimeError(f"No images found in {args.image_folder}")
        images = load_images_training_style(image_paths, img_size=args.img_size)
        images = images.to(device)
        frame_labels = [f"t{i}" for i in range(len(image_paths))]
        original_images = image_paths
        print(f"Loaded {images.shape[0]} frames (no GT)")

    # Run inference (same as eval_viser.py)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Running inference...")
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
    else:
        with torch.no_grad():
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in predictions:
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    # Export raw data - object_id combines checkpoint + data paths
    object_id, checkpoint_id, data_id = get_object_id(checkpoint_path, args.image_folder)

    # Extract data name from image_folder (e.g., "coat" from "coat_run_...")
    data_path_parts = Path(args.image_folder).parts
    data_name = None
    for part in data_path_parts:
        if "_run_" in part:
            data_name = part.split("_run_")[0]
            break
    if not data_name:
        data_name = Path(args.image_folder).name

    # Get experiment name from data_id (last component)
    experiment = Path(data_id).name

    frames_str = "_".join(str(i) for i in frame_indices) if frame_indices else "all"
    # Use "v" prefix for val-only runs to distinguish from train
    frames_prefix = "v" if args.val_only is not None else "f"
    run_name = f"{frames_prefix}{frames_str}"
    run_dir = output_dir / object_id / run_name

    print(f"\n{'='*60}")
    print(f"Exporting to {run_dir}...")
    print(f"{'='*60}")

    export_raw_data(
        pred_dict=predictions,
        output_dir=run_dir,
        gt_extrinsic=gt_extrinsic,
        gt_world_points=gt_world_points,
        gt_depths=gt_depths,
        frame_labels=frame_labels,
        gt_point_masks=gt_point_masks_np,
        original_images=original_images,
        gt_pose_enc=gt_pose_enc,
    )

    # Update index and runs
    # Display name from data (e.g., "Coat" from "coat_run_...")
    display_name = data_name.replace("_", " ").title() if data_name else experiment.replace("_", " ").title()
    update_index(output_dir, checkpoint_id)
    # Use first tag from args.tags, or data_name if none provided
    tag = args.tags[0] if args.tags else data_name
    update_runs(output_dir / object_id,
                checkpoint=checkpoint_path, data_path=args.image_folder, name=display_name)
    # Update model-level tags.json
    update_model_tags(output_dir, checkpoint_id, object_id, run_name, tag)

    # Cover image
    cover_path = output_dir / object_id / "gt_cover.png"
    if original_images:
        img = Image.open(original_images[0]).convert("RGB")
        img.thumbnail((200, 150), Image.LANCZOS)
        img.save(cover_path)

    print(f"\nExport complete! Data saved to: {run_dir}")


if __name__ == "__main__":
    main()
