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
import gzip
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

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
from vggt.utils.rotation import mat_to_quat as mat_to_quat_torch
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from data.dataset_util import resize_image_depth_and_intrinsic, crop_image_depth_and_intrinsic_by_pp

# =============================================================================
# DEFAULTS
# =============================================================================
CHECKPOINT = "/home/chuanruo/vggt_train/vggt_checkpoints/model.pt"
IMG_SIZE = 518
USE_LORA = False
FRAME_INDICES = [1, 3, 5, 7, 9, 17, 23]


# =============================================================================
# MODULE 1: IMAGE LOADING
# =============================================================================

def load_images_simple(image_paths: List[str], target_size: int = 518) -> torch.Tensor:
    """
    Load and preprocess images like test_co3d.py's load_and_preprocess_images.
    Simple resize preserving aspect ratio, no PP cropping or intrinsic adjustment.

    Args:
        image_paths: List of image file paths
        target_size: Target width (height computed to preserve aspect ratio)

    Returns:
        Tensor of shape (S, C, H, W)
    """
    images = []
    to_tensor = TF.ToTensor()

    for image_path in image_paths:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14  # Divisible by 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        # Center crop height if larger than target_size
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y:start_y + target_size, :]

        images.append(img)

    # Handle different shapes by padding
    shapes = set((img.shape[1], img.shape[2]) for img in images)
    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded = []
        for img in images:
            h_pad = max_h - img.shape[1]
            w_pad = max_w - img.shape[2]
            if h_pad > 0 or w_pad > 0:
                img = torch.nn.functional.pad(img, (w_pad//2, w_pad-w_pad//2, h_pad//2, h_pad-h_pad//2), value=1.0)
            padded.append(img)
        images = padded

    return torch.stack(images)


# =============================================================================
# MODULE 2: GT DATA LOADING FROM ANNOTATIONS
# =============================================================================

def load_annotations_file(annotations_dir: str, category: str, split: str) -> Optional[dict]:
    """Load annotation JSON/JGZ file."""
    patterns = [
        f"{category}_{split}.jgz",
        f"{category}.json",
        f"{category}_{split}.json",
    ]

    for pattern in patterns:
        path = os.path.join(annotations_dir, pattern)
        if os.path.exists(path):
            if path.endswith('.jgz'):
                with gzip.open(path, 'r') as f:
                    return json.loads(f.read())
            else:
                with open(path) as f:
                    return json.load(f)
    return None


def load_gt_from_annotations(
    annotations_dir: str,
    category: str,
    split: str = "train",
    sequence: str = None,
    frame_indices: List[int] = None,
    data_dir: str = None
) -> Tuple[Dict[str, Any], str, List[int]]:
    """
    Load all GT data from annotation files.

    Returns:
        gt_data: Dict with keys:
            - image_paths: List[str]
            - extrinsics: np.ndarray (S, 3, 4) - RAW extrinsics for camera pose metrics
            - intrinsics: np.ndarray (S, 3, 3) - original resolution intrinsics
            - depth_paths: List[str]
            - mask_paths: List[str]
            - depth_scales: List[float]
        seq_name: str
        indices: List[int]
    """
    anno = load_annotations_file(annotations_dir, category, split)
    if anno is None:
        return None, None, None

    # Get sequence
    seq_names = list(anno.keys())
    if sequence and sequence in anno:
        seq = sequence
    else:
        seq = seq_names[0]
        if len(seq_names) > 1:
            print(f"Multiple sequences: {seq_names}. Using '{seq}'.")

    seq_data = anno[seq]
    n_frames = len(seq_data)

    # Get frame indices
    if frame_indices is not None:
        indices = [i for i in frame_indices if i < n_frames]
    else:
        indices = list(range(n_frames))

    # Extract data for each frame
    image_paths = []
    depth_paths = []
    mask_paths = []
    extrinsics = []
    intrinsics = []
    depth_scales = []

    for i in indices:
        frame = seq_data[i]

        # Image path
        filepath = frame.get('filepath', frame.get('image_path', ''))
        if data_dir:
            if category in filepath:
                full_path = os.path.join(data_dir, filepath)
            else:
                full_path = os.path.join(data_dir, category, filepath)
        else:
            full_path = filepath
        image_paths.append(full_path)

        # Depth path
        depth_path = frame.get('depth_path', '')
        if depth_path and data_dir:
            if category in depth_path:
                depth_paths.append(os.path.join(data_dir, depth_path))
            else:
                depth_paths.append(os.path.join(data_dir, category, depth_path))
        else:
            depth_paths.append(None)

        # Mask path
        mask_path = frame.get('depth_mask_path', '')
        if mask_path and data_dir:
            if category in mask_path:
                mask_paths.append(os.path.join(data_dir, mask_path))
            else:
                mask_paths.append(os.path.join(data_dir, category, mask_path))
        else:
            mask_paths.append(None)

        # Depth scale
        depth_scales.append(frame.get('depth_scale_adjustment', 1.0))

        # Extrinsic
        if 'extri' in frame:
            extrinsics.append(np.array(frame['extri']))
        elif 'R' in frame and 'T' in frame:
            rot = np.array(frame['R'])
            trans = np.array(frame['T'])
            trans[:2] *= -1
            rot[:, :2] *= -1
            rot = rot.T
            extrinsics.append(np.hstack((rot, trans[:, None])))
        else:
            extrinsics.append(None)

        # Intrinsic
        if 'intri' in frame:
            intrinsics.append(np.array(frame['intri']))
        elif 'focal_length' in frame and 'principal_point' in frame:
            fl = np.array(frame['focal_length'])
            pp = np.array(frame['principal_point'])
            intrinsics.append(np.array([
                [fl[0], 0, pp[0]],
                [0, fl[1], pp[1]],
                [0, 0, 1]
            ], dtype=np.float32))
        else:
            intrinsics.append(None)

    # Stack arrays
    extrinsics = np.stack([e for e in extrinsics if e is not None]) if any(e is not None for e in extrinsics) else None
    intrinsics = np.stack([e for e in intrinsics if e is not None]) if any(e is not None for e in intrinsics) else None

    gt_data = {
        'image_paths': image_paths,
        'extrinsics': extrinsics,
        'intrinsics': intrinsics,
        'depth_paths': depth_paths,
        'mask_paths': mask_paths,
        'depth_scales': depth_scales,
    }

    return gt_data, seq, indices


def load_gt_depths_and_masks(
    depth_paths: List[str],
    mask_paths: List[str],
    depth_scales: List[float]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load GT depth maps and masks from paths.

    Returns:
        depths: (S, H, W) float32 array or None
        masks: (S, H, W) bool array or None
    """
    depths = []
    masks = []

    for depth_path, mask_path, scale in zip(depth_paths, mask_paths, depth_scales):
        if depth_path and os.path.exists(depth_path):
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img).astype(np.float32)
            depth = depth / 1000.0 * scale  # Convert to meters
            depths.append(depth)
        else:
            depths.append(None)

        if mask_path and os.path.exists(mask_path):
            mask_img = Image.open(mask_path)
            masks.append(np.array(mask_img).astype(bool))
        else:
            masks.append(None)

    depths = np.stack(depths) if all(d is not None for d in depths) else None
    masks = np.stack(masks) if all(m is not None for m in masks) else None

    return depths, masks


# =============================================================================
# MODULE 3: GT DATA PROCESSING (resize, world points, normalization)
# =============================================================================

def resize_gt_data(
    gt_depths: np.ndarray,
    gt_masks: np.ndarray,
    gt_intrinsics: np.ndarray,
    target_h: int,
    target_w: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resize GT depths, masks, and scale intrinsics to match target resolution.

    Args:
        gt_depths: (S, H_orig, W_orig)
        gt_masks: (S, H_orig, W_orig)
        gt_intrinsics: (S, 3, 3)
        target_h, target_w: target resolution

    Returns:
        resized_depths, resized_masks, scaled_intrinsics
    """
    from scipy.ndimage import zoom

    orig_h, orig_w = gt_depths.shape[1], gt_depths.shape[2]
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w

    # Resize depths
    resized_depths = zoom(gt_depths, (1, scale_h, scale_w), order=1)

    # Resize masks (nearest neighbor)
    resized_masks = zoom(gt_masks.astype(float), (1, scale_h, scale_w), order=0) > 0.5

    # Scale intrinsics
    scaled_intrinsics = gt_intrinsics.copy()
    scaled_intrinsics[:, 0, 0] *= scale_w  # fx
    scaled_intrinsics[:, 1, 1] *= scale_h  # fy
    scaled_intrinsics[:, 0, 2] *= scale_w  # cx
    scaled_intrinsics[:, 1, 2] *= scale_h  # cy

    return resized_depths, resized_masks, scaled_intrinsics


def compute_world_points_from_depth(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray
) -> np.ndarray:
    """
    Unproject depth maps to world coordinates.

    Args:
        depths: (S, H, W)
        intrinsics: (S, 3, 3)
        extrinsics: (S, 3, 4) world-to-camera

    Returns:
        world_points: (S, H, W, 3)
    """
    S, H, W = depths.shape
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    world_points = []
    for i in range(S):
        depth = depths[i]
        K = intrinsics[i]
        E = extrinsics[i]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Unproject to camera coordinates
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Transform to world coordinates
        R = E[:3, :3]
        t = E[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t

        cam_flat = cam_points.reshape(-1, 3)
        world_flat = cam_flat @ R_inv.T + t_inv
        world_points.append(world_flat.reshape(H, W, 3))

    return np.stack(world_points)


def compute_cam_points_from_world(
    world_points: np.ndarray,
    extrinsics: np.ndarray
) -> np.ndarray:
    """
    Project world points to camera coordinates.

    Args:
        world_points: (S, H, W, 3)
        extrinsics: (S, 3, 4) world-to-camera

    Returns:
        cam_points: (S, H, W, 3)
    """
    S = world_points.shape[0]
    cam_points = []

    for i in range(S):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        wp = world_points[i]  # (H, W, 3)
        cp = np.einsum('ij,hwj->hwi', R, wp) + t
        cam_points.append(cp)

    return np.stack(cam_points)


def normalize_gt_for_pose_encoding(
    gt_extrinsics: np.ndarray,
    gt_intrinsics: np.ndarray,
    gt_depths: np.ndarray,
    gt_world_points: np.ndarray,
    gt_masks: np.ndarray,
    image_hw: Tuple[int, int]
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Normalize GT data and compute pose encoding.

    Args:
        gt_extrinsics: (S, 3, 4)
        gt_intrinsics: (S, 3, 3)
        gt_depths: (S, H, W)
        gt_world_points: (S, H, W, 3)
        gt_masks: (S, H, W)
        image_hw: (H, W) of inference images

    Returns:
        gt_pose_enc: torch.Tensor (1, S, 9)
        norm_world_points: (S, H, W, 3)
        norm_depths: (S, H, W)
    """
    gt_ext_tensor = torch.from_numpy(gt_extrinsics).float().unsqueeze(0)
    gt_int_tensor = torch.from_numpy(gt_intrinsics).float().unsqueeze(0)
    gt_wp_tensor = torch.from_numpy(gt_world_points).float().unsqueeze(0)
    gt_mask_tensor = torch.from_numpy(gt_masks).float().unsqueeze(0)
    gt_depth_tensor = torch.from_numpy(gt_depths).float().unsqueeze(0).unsqueeze(-1)

    # Compute cam_points
    gt_cam_points = compute_cam_points_from_world(gt_world_points, gt_extrinsics)
    gt_cam_tensor = torch.from_numpy(gt_cam_points).float().unsqueeze(0)

    # Normalize
    norm_ext, norm_cam, norm_wp, norm_depths = normalize_camera_extrinsics_and_points_batch(
        extrinsics=gt_ext_tensor,
        cam_points=gt_cam_tensor,
        world_points=gt_wp_tensor,
        depths=gt_depth_tensor,
        point_masks=gt_mask_tensor,
        scale_by_points=True,
    )

    # Compute pose encoding
    H, W = image_hw
    gt_pose_enc = extri_intri_to_pose_encoding(norm_ext, gt_int_tensor, image_size_hw=(H, W))

    return gt_pose_enc, norm_wp[0].numpy(), norm_depths[0].squeeze(-1).numpy()


# =============================================================================
# MODULE 4: CAMERA POSE METRICS (RRA, RTA, AUC)
# =============================================================================

def compute_auc(errors: List[float], thresholds: List[int]) -> Dict[int, float]:
    """
    Compute AUC using histogram cumsum method (same as test_co3d.py).
    """
    if len(errors) == 0:
        return {t: 0.0 for t in thresholds}

    errors = np.array(errors)
    auc_results = {}

    for max_thresh in thresholds:
        bins = np.arange(max_thresh + 1)
        histogram, _ = np.histogram(errors, bins=bins)
        num_pairs = float(len(errors))
        normalized_histogram = histogram.astype(float) / num_pairs
        auc = np.mean(np.cumsum(normalized_histogram))
        auc_results[max_thresh] = float(auc)

    return auc_results


def compute_camera_pose_metrics(
    pred_extrinsics: np.ndarray,
    gt_extrinsics: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Compute camera pose metrics (RRA, RTA, AUC) over all pairs of views.
    Uses torch-based computation matching test_co3d.py exactly.

    Args:
        pred_extrinsics: (S, 3, 4) predicted camera extrinsics
        gt_extrinsics: (S, 3, 4) GT camera extrinsics (RAW, not normalized)

    Returns:
        Dict with camera pose metrics
    """
    N = len(pred_extrinsics)
    if N < 2:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_ext = torch.from_numpy(pred_extrinsics).double().to(device)
    gt_ext = torch.from_numpy(gt_extrinsics).double().to(device)

    # Add homogeneous row
    add_row = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float64).expand(N, 1, 4)
    pred_se3 = torch.cat((pred_ext, add_row), dim=1)
    gt_se3 = torch.cat((gt_ext, add_row), dim=1)

    # Build all pairs
    i1, i2 = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)

    # Compute relative poses
    relative_gt = gt_se3[i1].bmm(closed_form_inverse_se3(gt_se3[i2]))
    relative_pred = pred_se3[i1].bmm(closed_form_inverse_se3(pred_se3[i2]))

    # Rotation angle error using quaternions
    q_pred = mat_to_quat_torch(relative_pred[:, :3, :3])
    q_gt = mat_to_quat_torch(relative_gt[:, :3, :3])
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=1e-15)
    rra = torch.arccos((1 - 2 * loss_q).clamp(-1, 1)) * 180 / np.pi

    # Translation angle error
    t_pred = relative_pred[:, :3, 3]
    t_gt = relative_gt[:, :3, 3]
    t_pred_norm = t_pred / (torch.norm(t_pred, dim=1, keepdim=True) + 1e-15)
    t_gt_norm = t_gt / (torch.norm(t_gt, dim=1, keepdim=True) + 1e-15)
    loss_t = (1.0 - (t_pred_norm * t_gt_norm).sum(dim=1) ** 2).clamp(min=1e-15)
    rta = torch.acos((1 - loss_t).clamp(0, 1).sqrt()) * 180 / np.pi
    rta = torch.min(rta, (180 - rta).abs())

    # Convert to numpy
    rra_np = rra.cpu().numpy()
    rta_np = rta.cpu().numpy()
    pair_errors = np.maximum(rra_np, rta_np)

    # Compute AUC
    auc_results = compute_auc(pair_errors.tolist(), [3, 5, 10, 30])

    return {
        "num_pairs": int(len(pair_errors)),
        "rra_mean": round(float(rra_np.mean()), 4),
        "rta_mean": round(float(rta_np.mean()), 4),
        "pose_error_mean": round(float(pair_errors.mean()), 4),
        "rra_median": round(float(np.median(rra_np)), 4),
        "rta_median": round(float(np.median(rta_np)), 4),
        "pose_error_median": round(float(np.median(pair_errors)), 4),
        "auc_3": round(float(auc_results[3]), 4),
        "auc_5": round(float(auc_results[5]), 4),
        "auc_10": round(float(auc_results[10]), 4),
        "auc_30": round(float(auc_results[30]), 4),
    }


# =============================================================================
# MODULE 5: OTHER METRICS (depth, pointmap, chamfer, pose encoding)
# =============================================================================

def compute_pose_encoding_metrics(
    pred_pose_enc: np.ndarray,
    gt_pose_enc: torch.Tensor,
    frame_labels: List[str]
) -> Dict[str, Any]:
    """Compute pose encoding metrics (loss_T, loss_R, loss_FL)."""
    gt = gt_pose_enc[0].cpu().numpy() if hasattr(gt_pose_enc, 'cpu') else gt_pose_enc[0]
    pred = pred_pose_enc[0] if len(pred_pose_enc.shape) > 2 else pred_pose_enc
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()

    S = min(len(gt), len(pred))
    pose_per_frame = []

    for i in range(S):
        loss_T = float(np.abs(gt[i, :3] - pred[i, :3]).mean())
        loss_R = float(np.abs(gt[i, 3:7] - pred[i, 3:7]).mean())
        loss_FL = float(np.abs(gt[i, 7:] - pred[i, 7:]).mean())
        pose_per_frame.append({
            "frame": frame_labels[i] if i < len(frame_labels) else f"t{i}",
            "loss_T": round(loss_T, 6),
            "loss_R": round(loss_R, 6),
            "loss_FL": round(loss_FL, 6),
        })

    return {
        "per_frame": pose_per_frame,
        "mean_loss_T": round(float(np.mean([p["loss_T"] for p in pose_per_frame])), 6),
        "mean_loss_R": round(float(np.mean([p["loss_R"] for p in pose_per_frame])), 6),
        "mean_loss_FL": round(float(np.mean([p["loss_FL"] for p in pose_per_frame])), 6),
    }


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depths: np.ndarray,
    gt_masks: np.ndarray,
    frame_labels: List[str]
) -> Dict[str, Any]:
    """Compute depth MAE/RMSE metrics."""
    pred_depth_sq = pred_depth.squeeze(-1) if pred_depth.ndim == 4 else pred_depth
    gt_depths_sq = gt_depths.squeeze(-1) if gt_depths.ndim == 4 else gt_depths

    S = min(len(pred_depth_sq), len(gt_depths_sq))
    depth_per_frame = []

    for i in range(S):
        pred_d = pred_depth_sq[i]
        gt_d = gt_depths_sq[i]
        mask = gt_masks[i].astype(bool) if gt_masks is not None and i < len(gt_masks) else (gt_d > 0) & (pred_d > 0)

        if mask.sum() > 0:
            diff = np.abs(pred_d[mask] - gt_d[mask])
            mae = float(np.mean(diff))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            n_pixels = int(mask.sum())
        else:
            mae, rmse, n_pixels = 0.0, 0.0, 0

        depth_per_frame.append({
            "frame": frame_labels[i] if i < len(frame_labels) else f"t{i}",
            "depth_mae": round(mae, 6),
            "depth_rmse": round(rmse, 6),
            "n_pixels": n_pixels,
        })

    return {
        "per_frame": depth_per_frame,
        "mean_depth_mae": round(float(np.mean([d["depth_mae"] for d in depth_per_frame])), 6),
        "mean_depth_rmse": round(float(np.mean([d["depth_rmse"] for d in depth_per_frame])), 6),
    }


def compute_pointmap_metrics(
    pred_world_points: np.ndarray,
    gt_world_points: np.ndarray,
    gt_masks: np.ndarray,
    frame_labels: List[str]
) -> Dict[str, Any]:
    """Compute pointmap MAE/RMSE metrics."""
    S = min(len(pred_world_points), len(gt_world_points))
    pointmap_per_frame = []

    for i in range(S):
        pred_pts = pred_world_points[i]
        gt_pts = gt_world_points[i]
        mask = gt_masks[i].astype(bool) if gt_masks is not None and i < len(gt_masks) else np.ones(pred_pts.shape[:2], dtype=bool)

        if mask.sum() > 0:
            diff = np.linalg.norm(pred_pts[mask] - gt_pts[mask], axis=-1)
            mae = float(np.mean(diff))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            n_points = int(mask.sum())
        else:
            mae, rmse, n_points = 0.0, 0.0, 0

        pointmap_per_frame.append({
            "frame": frame_labels[i] if i < len(frame_labels) else f"t{i}",
            "pointmap_mae": round(mae, 6),
            "pointmap_rmse": round(rmse, 6),
            "n_points": n_points,
        })

    return {
        "per_frame": pointmap_per_frame,
        "mean_pointmap_mae": round(float(np.mean([p["pointmap_mae"] for p in pointmap_per_frame])), 6),
        "mean_pointmap_rmse": round(float(np.mean([p["pointmap_rmse"] for p in pointmap_per_frame])), 6),
    }


def compute_chamfer_distance(pred_points: np.ndarray, gt_points: np.ndarray, align: bool = True) -> Optional[Dict[str, float]]:
    """Compute Chamfer distance between point clouds."""
    pred_flat = pred_points.reshape(-1, 3)
    gt_flat = gt_points.reshape(-1, 3)

    # Filter invalid points
    pred_valid = np.isfinite(pred_flat).all(axis=1) & (np.abs(pred_flat).sum(axis=1) > 1e-8)
    gt_valid = np.isfinite(gt_flat).all(axis=1) & (np.abs(gt_flat).sum(axis=1) > 1e-8)
    pred_flat = pred_flat[pred_valid]
    gt_flat = gt_flat[gt_valid]

    if len(pred_flat) < 10 or len(gt_flat) < 10:
        return None

    # Subsample
    max_points = 50000
    if len(pred_flat) > max_points:
        pred_flat = pred_flat[np.random.choice(len(pred_flat), max_points, replace=False)]
    if len(gt_flat) > max_points:
        gt_flat = gt_flat[np.random.choice(len(gt_flat), max_points, replace=False)]

    scale = 1.0
    if align:
        pred_centroid = np.mean(pred_flat, axis=0)
        gt_centroid = np.mean(gt_flat, axis=0)
        pred_centered = pred_flat - pred_centroid
        gt_centered = gt_flat - gt_centroid
        pred_scale = np.sqrt(np.mean(np.sum(pred_centered ** 2, axis=1)))
        gt_scale = np.sqrt(np.mean(np.sum(gt_centered ** 2, axis=1)))
        if pred_scale > 1e-8:
            scale = float(gt_scale / pred_scale)
        pred_aligned = pred_centered * scale + gt_centroid
    else:
        pred_aligned = pred_flat

    if HAS_SCIPY:
        gt_tree = cKDTree(gt_flat)
        pred_tree = cKDTree(pred_aligned)
        dist_pred_to_gt, _ = gt_tree.query(pred_aligned, k=1)
        dist_gt_to_pred, _ = pred_tree.query(gt_flat, k=1)
    else:
        # Fallback
        def nn_dist(query, target):
            dists = np.zeros(len(query))
            for i in range(0, len(query), 1000):
                end = min(i + 1000, len(query))
                diff = query[i:end, None, :] - target[None, :, :]
                dists[i:end] = np.min(np.linalg.norm(diff, axis=2), axis=1)
            return dists
        dist_pred_to_gt = nn_dist(pred_aligned, gt_flat)
        dist_gt_to_pred = nn_dist(gt_flat, pred_aligned)

    accuracy = float(np.mean(dist_pred_to_gt))
    completeness = float(np.mean(dist_gt_to_pred))

    return {
        "chamfer_accuracy": round(accuracy, 6),
        "chamfer_completeness": round(completeness, 6),
        "chamfer_overall": round((accuracy + completeness) / 2, 6),
        "alignment_scale": round(float(scale), 4),
        "num_pred_points": int(len(pred_aligned)),
        "num_gt_points": int(len(gt_flat)),
    }


def compute_all_metrics(
    pred_extrinsics: np.ndarray,
    pred_depth: np.ndarray,
    pred_world_points: np.ndarray,
    pred_pose_enc: np.ndarray,
    gt_extrinsics: np.ndarray,  # RAW extrinsics
    gt_depths: np.ndarray,      # Normalized depths
    gt_world_points: np.ndarray,  # Normalized world points
    gt_masks: np.ndarray,
    gt_pose_enc: torch.Tensor,
    frame_labels: List[str]
) -> Dict[str, Any]:
    """Compute all metrics."""
    metrics = {"num_frames": len(pred_extrinsics)}

    # Camera pose metrics (uses RAW gt_extrinsics)
    if gt_extrinsics is not None and len(gt_extrinsics) >= 2:
        camera_pose = compute_camera_pose_metrics(pred_extrinsics, gt_extrinsics)
        if camera_pose:
            metrics["camera_pose"] = camera_pose

    # Pose encoding metrics
    if gt_pose_enc is not None and pred_pose_enc is not None:
        metrics["pose"] = compute_pose_encoding_metrics(pred_pose_enc, gt_pose_enc, frame_labels)

    # Depth metrics
    if gt_depths is not None:
        metrics["depth"] = compute_depth_metrics(pred_depth, gt_depths, gt_masks, frame_labels)

    # Pointmap metrics
    if gt_world_points is not None:
        metrics["pointmap"] = compute_pointmap_metrics(pred_world_points, gt_world_points, gt_masks, frame_labels)

    # Chamfer metrics
    if gt_world_points is not None:
        all_pred = pred_world_points.reshape(-1, 3)
        all_gt = gt_world_points.reshape(-1, 3)
        chamfer = compute_chamfer_distance(all_pred, all_gt, align=True)
        if chamfer:
            metrics["chamfer"] = chamfer

    return metrics


# =============================================================================
# MODULE 6: EXPORT
# =============================================================================

def export_raw_data(
    pred_dict: Dict[str, np.ndarray],
    output_dir: Path,
    gt_extrinsics: np.ndarray = None,
    gt_world_points: np.ndarray = None,
    gt_depths: np.ndarray = None,
    gt_masks: np.ndarray = None,
    gt_pose_enc: torch.Tensor = None,
    frame_labels: List[str] = None,
    original_images: List[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Export predictions and compute metrics."""
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    images = pred_dict["images"]
    world_points = pred_dict["world_points"]
    world_points_conf = pred_dict["world_points_conf"]
    depth = pred_dict["depth"]
    depth_conf = pred_dict["depth_conf"]
    extrinsics = pred_dict["extrinsic"]
    intrinsics = pred_dict["intrinsic"]

    S, C, H, W = images.shape
    labels = frame_labels if frame_labels else [f"t{i}" for i in range(S)]

    print(f"  Image shape: {S} x {C} x {H} x {W}")

    # Scene center
    all_points = world_points.reshape(-1, 3)
    scene_center = np.mean(all_points, axis=0)
    print(f"  Scene center: {scene_center}")
    print(f"  Depth range: {depth.min():.3f} - {depth.max():.3f}")
    print(f"  Confidence range: {world_points_conf.min():.3f} - {world_points_conf.max():.3f}")

    if not dry_run:
        # Create directories and save data
        for subdir in ["images", "depths", "pointmaps", "confidence", "depth_conf"]:
            (output_dir / subdir).mkdir(exist_ok=True)

        for i in range(S):
            img = (images[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(output_dir / "images" / f"view_{i:03d}.png")
            depth[i].squeeze(-1).astype(np.float32).tofile(output_dir / "depths" / f"view_{i:03d}.bin")
            (world_points[i] - scene_center).astype(np.float32).tofile(output_dir / "pointmaps" / f"view_{i:03d}.bin")
            world_points_conf[i].astype(np.float32).tofile(output_dir / "confidence" / f"view_{i:03d}.bin")
            depth_conf[i].astype(np.float32).tofile(output_dir / "depth_conf" / f"view_{i:03d}.bin")

        # GT data
        if gt_world_points is not None:
            (output_dir / "gt_pointmaps").mkdir(exist_ok=True)
            for i in range(min(S, len(gt_world_points))):
                (gt_world_points[i] - scene_center).astype(np.float32).tofile(output_dir / "gt_pointmaps" / f"view_{i:03d}.bin")

        if gt_depths is not None:
            (output_dir / "gt_depths").mkdir(exist_ok=True)
            gt_d = gt_depths.squeeze(-1) if gt_depths.ndim == 4 else gt_depths
            for i in range(min(S, len(gt_d))):
                gt_d[i].astype(np.float32).tofile(output_dir / "gt_depths" / f"view_{i:03d}.bin")

        if gt_masks is not None:
            (output_dir / "gt_masks").mkdir(exist_ok=True)
            for i in range(min(S, len(gt_masks))):
                gt_masks[i].astype(np.uint8).tofile(output_dir / "gt_masks" / f"view_{i:03d}.bin")

        print(f"  Wrote {S} frames")

    # Cameras JSON
    cam_to_world = closed_form_inverse_se3(extrinsics)[:, :3, :].copy()
    cam_to_world[..., -1] -= scene_center

    pred_cameras = []
    for i in range(S):
        fov_deg = float(np.degrees(2 * np.arctan2(H / 2, 1.1 * H)))
        pred_cameras.append({
            "view_id": labels[i],
            "position": cam_to_world[i, :, 3].tolist(),
            "matrix": cam_to_world[i].tolist(),
            "extrinsic": extrinsics[i].tolist(),
            "intrinsic": intrinsics[i].tolist(),
            "fov_deg": round(fov_deg, 1),
            "aspect": round(W / H, 3),
            "source_image": original_images[i] if original_images and i < len(original_images) else None
        })

    gt_cameras = []
    if gt_extrinsics is not None:
        gt_c2w = closed_form_inverse_se3(gt_extrinsics)[:, :3, :]
        gt_c2w[..., -1] -= scene_center
        for i in range(min(S, len(gt_extrinsics))):
            gt_cameras.append({
                "view_id": labels[i],
                "position": gt_c2w[i, :, 3].tolist(),
                "matrix": gt_c2w[i].tolist(),
                "extrinsic": gt_extrinsics[i].tolist(),
            })

    cameras_data = {
        "num_frames": S,
        "height": H,
        "width": W,
        "scene_center": scene_center.tolist(),
        "pred_cameras": pred_cameras,
        "gt_cameras": gt_cameras if gt_cameras else None,
    }

    if not dry_run:
        with open(output_dir / "cameras.json", 'w') as f:
            json.dump(cameras_data, f, indent=2)

    # Compute metrics
    metrics = compute_all_metrics(
        pred_extrinsics=extrinsics,
        pred_depth=depth,
        pred_world_points=world_points,
        pred_pose_enc=pred_dict.get("pose_enc"),
        gt_extrinsics=gt_extrinsics,
        gt_depths=gt_depths,
        gt_world_points=gt_world_points,
        gt_masks=gt_masks,
        gt_pose_enc=gt_pose_enc,
        frame_labels=labels,
    )
    metrics["conf_min"] = float(world_points_conf.min())
    metrics["conf_max"] = float(world_points_conf.max())
    metrics["mean_depth"] = float(depth.mean())

    if dry_run:
        print(f"\n=== DRY RUN METRICS ===")
        if "camera_pose" in metrics:
            cp = metrics["camera_pose"]
            print(f"Camera Pose: AUC@30={cp['auc_30']:.4f}, AUC@10={cp['auc_10']:.4f}")
            print(f"             RRA={cp['rra_mean']:.4f}, RTA={cp['rta_mean']:.4f}")
        if "pose" in metrics:
            p = metrics["pose"]
            print(f"Pose Enc: T={p['mean_loss_T']:.4f}, R={p['mean_loss_R']:.4f}, FL={p['mean_loss_FL']:.4f}")
        if "depth" in metrics:
            print(f"Depth MAE: {metrics['depth']['mean_depth_mae']:.4f}")
        if "pointmap" in metrics:
            print(f"Pointmap MAE: {metrics['pointmap']['mean_pointmap_mae']:.4f}")
    else:
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Wrote metrics.json")

    return metrics


# =============================================================================
# MODULE 7: INDEX/TAGS MANAGEMENT
# =============================================================================

def get_object_id(checkpoint_path: str, data_path: str) -> Tuple[str, str, str]:
    """Extract object_id from checkpoint + data paths."""
    ckpt = Path(checkpoint_path).resolve()
    ckpt_name = ckpt.stem
    exp_name = None
    for part in ckpt.parts:
        if part.startswith("exp"):
            exp_name = part
            break
    checkpoint_id = f"{exp_name}_ckpts_{ckpt_name}" if exp_name else ckpt_name

    data = Path(data_path).resolve()
    data_str = str(data)
    for base in ["/home/chuanruo/TGGT/out/", "/home/chuanruo/TGGT/out"]:
        if data_str.startswith(base):
            data_str = data_str[len(base):]
            break
    data_parts = [p for p in data_str.strip("/").split("/") if p != "subsets"]
    data_id = "_".join(data_parts)

    return f"{checkpoint_id}/{data_id}", checkpoint_id, data_id


def update_index(output_dir: Path, checkpoint_id: str):
    """Add checkpoint ID to index.json."""
    index_path = output_dir / "index.json"
    index = json.load(open(index_path)) if index_path.exists() else []
    if checkpoint_id not in index:
        index.append(checkpoint_id)
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


def update_runs(object_dir: Path, checkpoint: str = None, data_path: str = None, name: str = None):
    """Update runs.json for an object."""
    runs_path = object_dir / "runs.json"
    runs = json.load(open(runs_path)) if runs_path.exists() else {}
    if name: runs["name"] = name
    if checkpoint: runs["checkpoint"] = checkpoint
    if data_path: runs["data_path"] = data_path
    with open(runs_path, 'w') as f:
        json.dump(runs, f, indent=2)


def update_model_tags(output_dir: Path, checkpoint_id: str, object_id: str, run_name: str, tag: str):
    """Update tags.json at checkpoint level."""
    tags_path = output_dir / checkpoint_id / "tags.json"
    tags_path.parent.mkdir(parents=True, exist_ok=True)
    tags_data = json.load(open(tags_path)) if tags_path.exists() else {"tags": {}}
    if "tags" not in tags_data:
        tags_data["tags"] = {}

    rel_object = object_id[len(checkpoint_id):].strip("/")
    run_path = f"{rel_object}/{run_name}"

    if tag not in tags_data["tags"]:
        tags_data["tags"][tag] = []
    if run_path not in tags_data["tags"][tag]:
        tags_data["tags"][tag].append(run_path)

    with open(tags_path, 'w') as f:
        json.dump(tags_data, f, indent=2)


# =============================================================================
# MODULE 8: MAIN PROCESSING
# =============================================================================

def process_category(
    model,
    device,
    image_folder: str,
    annotations_dir: str,
    output_dir: Path,
    checkpoint_path: str,
    frame_indices: List[int],
    args,
    category_name: str = None,
    sequence_name: str = None,
):
    """Process a single category and export results."""
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    has_annotations = os.path.isdir(annotations_dir)
    seq = sequence_name if sequence_name else args.sequence
    cat = category_name if category_name else os.path.basename(image_folder.rstrip('/'))

    # Initialize GT data
    gt_extrinsics_raw = None  # Keep raw for camera pose metrics
    gt_intrinsics = None
    gt_depths = None
    gt_world_points = None
    gt_masks = None
    gt_pose_enc = None
    frame_labels = None
    original_images = []

    if has_annotations:
        print("Loading with simple preprocessing (like test_co3d.py)...")
        val_only = args.val_only is not None
        split = "test" if val_only else "train"

        # Step 1: Load GT data from annotations
        gt_data, seq_used, indices_used = load_gt_from_annotations(
            annotations_dir, cat, split=split, sequence=seq,
            frame_indices=frame_indices, data_dir=image_folder
        )
        if gt_data is None:
            raise RuntimeError(f"Could not load annotations for {cat} from {annotations_dir}")

        image_paths = gt_data['image_paths']
        gt_extrinsics_raw = gt_data['extrinsics']  # RAW - keep for camera pose metrics
        gt_intrinsics = gt_data['intrinsics']

        print(f"Sequence: {seq_used}, loading {len(image_paths)} frames at indices {indices_used}")

        # Step 2: Load images
        images = load_images_simple(image_paths, target_size=args.img_size).to(device)
        S, C, H, W = images.shape
        print(f"Image shape: {images.shape}")

        # Step 3: Load GT depths and masks
        gt_depths_orig, gt_masks_orig = load_gt_depths_and_masks(
            gt_data['depth_paths'],
            gt_data['mask_paths'],
            gt_data['depth_scales']
        )

        # Step 4: Resize GT to match inference resolution
        if gt_depths_orig is not None and gt_masks_orig is not None and gt_intrinsics is not None:
            print(f"Loaded GT depths: {gt_depths_orig.shape}, masks: {gt_masks_orig.shape}")
            gt_depths, gt_masks, gt_intrinsics_scaled = resize_gt_data(
                gt_depths_orig, gt_masks_orig, gt_intrinsics, H, W
            )
            print(f"Resized to: depths {gt_depths.shape}, masks {gt_masks.shape}")

            # Step 5: Compute world points from resized data
            gt_world_points = compute_world_points_from_depth(gt_depths, gt_intrinsics_scaled, gt_extrinsics_raw)
            print(f"Computed GT world points: {gt_world_points.shape}")

            # Step 6: Normalize and compute pose encoding
            gt_pose_enc, gt_world_points, gt_depths = normalize_gt_for_pose_encoding(
                gt_extrinsics_raw, gt_intrinsics_scaled, gt_depths, gt_world_points, gt_masks, (H, W)
            )
            print(f"Computed gt_pose_enc: {gt_pose_enc.shape}")

        frame_prefix = "v" if val_only else "t"
        frame_labels = [f"{frame_prefix}{i}" for i in indices_used]
        frame_indices_used = indices_used
        original_images = image_paths

    else:
        print("Flat folder — loading with simple preprocessing...")
        image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if args.max_images:
            image_paths = image_paths[:args.max_images]
        if not image_paths:
            raise RuntimeError(f"No images found in {image_folder}")
        images = load_images_simple(image_paths, target_size=args.img_size).to(device)
        frame_labels = [f"t{i}" for i in range(len(image_paths))]
        original_images = image_paths
        frame_indices_used = frame_indices
        print(f"Loaded {images.shape[0]} frames (no GT)")

    # Run inference
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

    # Export
    object_id, checkpoint_id, data_id = get_object_id(checkpoint_path, image_folder)
    if category_name:
        object_id = f"{object_id}_{category_name}"
        data_id = f"{data_id}_{category_name}"

    data_name = category_name or Path(image_folder).name
    frames_str = "_".join(str(i) for i in frame_indices_used) if frame_indices_used else "all"
    frames_prefix = "v" if args.val_only is not None else "f"
    run_name = f"{frames_prefix}{frames_str}"
    run_dir = output_dir / object_id / run_name

    print(f"\n{'='*60}")
    print(f"Exporting to {run_dir}...")
    print(f"{'='*60}")

    export_raw_data(
        pred_dict=predictions,
        output_dir=run_dir,
        gt_extrinsics=gt_extrinsics_raw,  # RAW extrinsics for camera pose
        gt_world_points=gt_world_points,
        gt_depths=gt_depths,
        gt_masks=gt_masks,
        gt_pose_enc=gt_pose_enc,
        frame_labels=frame_labels,
        original_images=original_images,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        update_index(output_dir, checkpoint_id)
        tag = args.tags[0] if args.tags else data_name
        update_runs(output_dir / object_id, checkpoint=checkpoint_path, data_path=image_folder, name=data_name)
        update_model_tags(output_dir, checkpoint_id, object_id, run_name, tag)

        if original_images:
            cover_path = output_dir / object_id / "gt_cover.png"
            img = Image.open(original_images[0]).convert("RGB")
            img.thumbnail((200, 150), Image.LANCZOS)
            img.save(cover_path)

        print(f"\nExport complete! Data saved to: {run_dir}")


def is_multi_category_dir(path):
    """Check if directory contains multiple categories."""
    path = Path(path)
    if not path.is_dir():
        return False, []
    categories = []
    for subdir in sorted(path.iterdir()):
        if subdir.is_dir():
            has_sequence = any(seq.is_dir() and seq.name.isdigit() for seq in subdir.iterdir())
            if has_sequence:
                categories.append(subdir.name)
    return len(categories) > 1, categories


# =============================================================================
# MAIN
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default=None)
parser.add_argument("--anno_dir", type=str, default=None)
parser.add_argument("--co3d_dir", type=str, default=None)
parser.add_argument("--co3d_anno_dir", type=str, default=None)
parser.add_argument("--category", type=str, default=None)
parser.add_argument("--output", type=str, default="results")
parser.add_argument("--sequence", type=str, default=None)
parser.add_argument("--max_images", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--val", type=int, nargs="*", default=None)
parser.add_argument("--val-only", type=int, nargs="*", default=None)
parser.add_argument("--epoch", type=int, default=0)
parser.add_argument("--tags", type=str, nargs="*", default=None)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--frames", type=int, nargs="*", default=None)
parser.add_argument("--num_frames", type=int, default=10)
parser.add_argument("--dry_run", action="store_true")


def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame_indices = args.frames if args.frames else FRAME_INDICES
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint if args.checkpoint else CHECKPOINT
    print(f"Loading model from {checkpoint_path}...")
    model = VGGT()
    if USE_LORA:
        model.apply_lora()

    if checkpoint_path.endswith("vggt_checkpoints/model.pt"):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        current_state = model.state_dict()
        state = {k: v for k, v in state.items() if k in current_state and v.shape == current_state[k].shape}
        model.load_state_dict(state, strict=False)
    model.eval().to(device)

    if args.co3d_dir and args.co3d_anno_dir and args.category:
        process_category(model, device, args.co3d_dir, args.co3d_anno_dir, output_dir, checkpoint_path, frame_indices, args, category_name=args.category)
    elif args.image_folder:
        is_multi, categories = is_multi_category_dir(args.image_folder)
        annotations_dir = args.anno_dir if args.anno_dir else args.image_folder.rstrip("/") + "_annotations"

        if is_multi and os.path.isdir(annotations_dir):
            cats = [args.category] if args.category else categories
            for i, cat in enumerate(cats):
                print(f"\n{'#'*60}\nProcessing {i+1}/{len(cats)}: {cat}\n{'#'*60}")
                try:
                    process_category(model, device, args.image_folder, annotations_dir, output_dir, checkpoint_path, frame_indices, args, category_name=cat, sequence_name=f"{cat}_000")
                except Exception as e:
                    print(f"Error processing {cat}: {e}")
            print(f"\n{'='*60}\nAll categories processed!\n{'='*60}")
        else:
            process_category(model, device, args.image_folder, annotations_dir, output_dir, checkpoint_path, frame_indices, args)
    else:
        raise ValueError("Must provide --image_folder or (--co3d_dir, --co3d_anno_dir, --category)")


if __name__ == "__main__":
    main()
