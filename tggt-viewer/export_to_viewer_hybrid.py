#!/usr/bin/env python3
"""
HYBRID VERSION:
- Image loading: Simple preprocessing (like current version / test_co3d.py)
- GT processing: Training pipeline (load_co3d_batch, get_gt_cameras)

This tests whether the difference in metrics is due to image loading or GT processing.
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
# DEFAULTS
# =============================================================================
CHECKPOINT = "/home/chuanruo/vggt_train/vggt_checkpoints/model.pt"
IMG_SIZE = 518
USE_LORA = False
FRAME_INDICES = [1, 3, 5, 7, 9, 17, 23]


# =============================================================================
# IMAGE LOADING: SIMPLE (from current version)
# =============================================================================

def load_images_simple(image_paths: List[str], target_size: int = 518) -> torch.Tensor:
    """
    Simple image loading like test_co3d.py.
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
        new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y:start_y + target_size, :]

        images.append(img)

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
# GT LOADING: TRAINING PIPELINE (from export_to_viewer_training.py)
# =============================================================================

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


# =============================================================================
# CAMERA POSE METRICS
# =============================================================================

def rotation_matrix_to_angle(R):
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle_rad)


def compute_relative_rotation_error(R_pred_i, R_pred_j, R_gt_i, R_gt_j):
    R_rel_pred = R_pred_j @ R_pred_i.T
    R_rel_gt = R_gt_j @ R_gt_i.T
    R_error = R_rel_pred @ R_rel_gt.T
    return rotation_matrix_to_angle(R_error)


def compute_relative_translation_error(t_pred_i, t_pred_j, t_gt_i, t_gt_j):
    t_rel_pred = t_pred_j - t_pred_i
    t_rel_gt = t_gt_j - t_gt_i
    norm_pred = np.linalg.norm(t_rel_pred)
    norm_gt = np.linalg.norm(t_rel_gt)
    if norm_pred < 1e-8 or norm_gt < 1e-8:
        return 0.0
    t_rel_pred = t_rel_pred / norm_pred
    t_rel_gt = t_rel_gt / norm_gt
    cos_angle = np.clip(np.dot(t_rel_pred, t_rel_gt), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_auc(errors, thresholds):
    if len(errors) == 0:
        return {t: 0.0 for t in thresholds}
    errors = np.array(errors)
    auc_results = {}
    for max_thresh in thresholds:
        sweep = np.linspace(0, max_thresh, 100)
        accuracies = [np.mean(errors < tau) for tau in sweep]
        auc = np.trapz(accuracies, sweep) / max_thresh
        auc_results[max_thresh] = float(auc)
    return auc_results


def compute_camera_pose_metrics(pred_extrinsics, gt_extrinsics):
    N = len(pred_extrinsics)
    if N < 2:
        return None

    R_pred = pred_extrinsics[:, :3, :3]
    t_pred = pred_extrinsics[:, :3, 3]
    R_gt = gt_extrinsics[:, :3, :3]
    t_gt = gt_extrinsics[:, :3, 3]

    rra_errors, rta_errors, pair_errors = [], [], []
    for i, j in combinations(range(N), 2):
        rra = compute_relative_rotation_error(R_pred[i], R_pred[j], R_gt[i], R_gt[j])
        rta = compute_relative_translation_error(t_pred[i], t_pred[j], t_gt[i], t_gt[j])
        rra_errors.append(rra)
        rta_errors.append(rta)
        pair_errors.append(max(rra, rta))

    auc_results = compute_auc(pair_errors, [3, 5, 10, 30])
    return {
        "num_pairs": int(len(pair_errors)),
        "rra_mean": round(float(np.mean(rra_errors)), 4),
        "rta_mean": round(float(np.mean(rta_errors)), 4),
        "pose_error_mean": round(float(np.mean(pair_errors)), 4),
        "auc_3": round(float(auc_results[3]), 4),
        "auc_5": round(float(auc_results[5]), 4),
        "auc_10": round(float(auc_results[10]), 4),
        "auc_30": round(float(auc_results[30]), 4),
    }


# =============================================================================
# CHAMFER DISTANCE
# =============================================================================

def compute_chamfer_distance(pred_points, gt_points, align=True):
    pred_flat = pred_points.reshape(-1, 3)
    gt_flat = gt_points.reshape(-1, 3)

    pred_valid = np.isfinite(pred_flat).all(axis=1) & (np.abs(pred_flat).sum(axis=1) > 1e-8)
    gt_valid = np.isfinite(gt_flat).all(axis=1) & (np.abs(gt_flat).sum(axis=1) > 1e-8)
    pred_flat = pred_flat[pred_valid]
    gt_flat = gt_flat[gt_valid]

    if len(pred_flat) < 10 or len(gt_flat) < 10:
        return None

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
            scale = gt_scale / pred_scale
        pred_aligned = pred_centered * scale + gt_centroid
    else:
        pred_aligned = pred_flat

    if HAS_SCIPY:
        gt_tree = cKDTree(gt_flat)
        pred_tree = cKDTree(pred_aligned)
        dist_pred_to_gt, _ = gt_tree.query(pred_aligned, k=1)
        dist_gt_to_pred, _ = pred_tree.query(gt_flat, k=1)
    else:
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
    }


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(
    pred_extrinsics, pred_depth, pred_world_points, pred_intrinsics,
    gt_extrinsic=None, gt_depths=None, gt_world_points=None, gt_point_masks=None,
    frame_labels=None, H=None, W=None, pred_pose_enc=None, gt_pose_enc=None,
):
    S = len(pred_extrinsics)
    labels = frame_labels if frame_labels else [f"t{i}" for i in range(S)]
    metrics = {"num_frames": S, "conf_threshold": 5.0}

    # Pose encoding metrics
    if gt_pose_enc is not None and pred_pose_enc is not None:
        pose_per_frame = []
        gt = gt_pose_enc[0].cpu().numpy() if hasattr(gt_pose_enc, 'cpu') else gt_pose_enc[0]
        pred = pred_pose_enc[0] if len(pred_pose_enc.shape) > 2 else pred_pose_enc
        if hasattr(pred, 'cpu'):
            pred = pred.cpu().numpy()

        for i in range(min(S, len(gt))):
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
            "mean_loss_T": round(float(np.mean([p["loss_T"] for p in pose_per_frame])), 6),
            "mean_loss_R": round(float(np.mean([p["loss_R"] for p in pose_per_frame])), 6),
            "mean_loss_FL": round(float(np.mean([p["loss_FL"] for p in pose_per_frame])), 6),
        }

    # Depth metrics
    if gt_depths is not None and len(gt_depths) > 0:
        depth_per_frame = []
        gt_depths_sq = gt_depths.squeeze(-1) if gt_depths.ndim == 4 else gt_depths
        pred_depth_sq = pred_depth.squeeze(-1) if pred_depth.ndim == 4 else pred_depth

        for i in range(min(S, len(gt_depths_sq))):
            pred_d = pred_depth_sq[i]
            gt_d = gt_depths_sq[i]
            mask = gt_point_masks[i].astype(bool) if gt_point_masks is not None and i < len(gt_point_masks) else (gt_d > 0) & (pred_d > 0)

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
            "mean_depth_mae": round(float(np.mean([d["depth_mae"] for d in depth_per_frame])), 6),
            "mean_depth_rmse": round(float(np.mean([d["depth_rmse"] for d in depth_per_frame])), 6),
        }

    # Pointmap metrics
    if gt_world_points is not None and len(gt_world_points) > 0:
        pointmap_per_frame = []
        for i in range(min(S, len(gt_world_points))):
            pred_pts = pred_world_points[i]
            gt_pts = gt_world_points[i]
            mask = gt_point_masks[i].astype(bool) if gt_point_masks is not None and i < len(gt_point_masks) else np.ones(pred_pts.shape[:2], dtype=bool)

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
            "mean_pointmap_mae": round(float(np.mean([p["pointmap_mae"] for p in pointmap_per_frame])), 6),
            "mean_pointmap_rmse": round(float(np.mean([p["pointmap_rmse"] for p in pointmap_per_frame])), 6),
        }

    # Camera pose metrics
    if gt_extrinsic is not None and len(gt_extrinsic) >= 2:
        camera_pose = compute_camera_pose_metrics(pred_extrinsics, gt_extrinsic)
        if camera_pose:
            metrics["camera_pose"] = camera_pose

    # Chamfer metrics
    if gt_world_points is not None:
        all_pred = pred_world_points.reshape(-1, 3)
        all_gt = gt_world_points.reshape(-1, 3)
        chamfer = compute_chamfer_distance(all_pred, all_gt, align=True)
        if chamfer:
            metrics["chamfer"] = chamfer

    return metrics


# =============================================================================
# EXPORT
# =============================================================================

def export_raw_data(
    pred_dict, output_dir: Path,
    gt_extrinsic=None, gt_world_points=None, gt_depths=None,
    frame_labels=None, gt_point_masks=None, original_images=None, gt_pose_enc=None,
):
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

    scene_center = np.mean(world_points.reshape(-1, 3), axis=0)
    cam_to_world_mat = closed_form_inverse_se3(extrinsics)
    cam_to_world = cam_to_world_mat[:, :3, :].copy()
    cam_to_world[..., -1] -= scene_center

    print(f"  Scene center: {scene_center}")
    print(f"  Depth range: {depth.min():.3f} - {depth.max():.3f}")
    print(f"  Confidence range: {world_points_conf.min():.3f} - {world_points_conf.max():.3f}")

    for subdir in ["images", "depths", "pointmaps", "confidence", "depth_conf"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    for i in range(S):
        img = (images[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(output_dir / "images" / f"view_{i:03d}.png")
        depth[i].squeeze(-1).astype(np.float32).tofile(output_dir / "depths" / f"view_{i:03d}.bin")
        (world_points[i] - scene_center).astype(np.float32).tofile(output_dir / "pointmaps" / f"view_{i:03d}.bin")
        world_points_conf[i].astype(np.float32).tofile(output_dir / "confidence" / f"view_{i:03d}.bin")
        depth_conf[i].astype(np.float32).tofile(output_dir / "depth_conf" / f"view_{i:03d}.bin")

    print(f"  Wrote {S} frames")

    if gt_world_points is not None:
        (output_dir / "gt_pointmaps").mkdir(exist_ok=True)
        for i in range(min(S, len(gt_world_points))):
            (gt_world_points[i] - scene_center).astype(np.float32).tofile(output_dir / "gt_pointmaps" / f"view_{i:03d}.bin")

    if gt_depths is not None:
        (output_dir / "gt_depths").mkdir(exist_ok=True)
        gt_d = gt_depths.squeeze(-1) if gt_depths.ndim == 4 else gt_depths
        for i in range(min(S, len(gt_d))):
            gt_d[i].astype(np.float32).tofile(output_dir / "gt_depths" / f"view_{i:03d}.bin")

    if gt_point_masks is not None:
        (output_dir / "gt_masks").mkdir(exist_ok=True)
        for i in range(min(S, len(gt_point_masks))):
            gt_point_masks[i].astype(np.uint8).tofile(output_dir / "gt_masks" / f"view_{i:03d}.bin")

    # Cameras JSON
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
        })

    gt_cameras = []
    if gt_extrinsic is not None:
        gt_c2w = closed_form_inverse_se3(gt_extrinsic)[:, :3, :]
        gt_c2w[..., -1] -= scene_center
        for i in range(min(S, len(gt_extrinsic))):
            gt_cameras.append({
                "view_id": labels[i],
                "position": gt_c2w[i, :, 3].tolist(),
                "matrix": gt_c2w[i].tolist(),
                "extrinsic": gt_extrinsic[i].tolist(),
            })

    cameras_data = {
        "num_frames": S, "height": H, "width": W,
        "scene_center": scene_center.tolist(),
        "pred_cameras": pred_cameras,
        "gt_cameras": gt_cameras if gt_cameras else None,
    }

    with open(output_dir / "cameras.json", 'w') as f:
        json.dump(cameras_data, f, indent=2)

    # Metrics
    pred_pose_enc = pred_dict.get("pose_enc")
    metrics = compute_metrics(
        pred_extrinsics=extrinsics, pred_depth=depth, pred_world_points=world_points,
        pred_intrinsics=intrinsics, gt_extrinsic=gt_extrinsic, gt_depths=gt_depths,
        gt_world_points=gt_world_points, gt_point_masks=gt_point_masks,
        frame_labels=labels, H=H, W=W, pred_pose_enc=pred_pose_enc, gt_pose_enc=gt_pose_enc,
    )
    metrics["conf_min"] = float(world_points_conf.min())
    metrics["conf_max"] = float(world_points_conf.max())
    metrics["mean_depth"] = float(depth.mean())
    metrics["split"] = "train"

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Wrote metrics.json")


# =============================================================================
# INDEX/TAGS
# =============================================================================

def get_object_id(checkpoint_path: str, data_path: str):
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
    index_path = output_dir / "index.json"
    index = json.load(open(index_path)) if index_path.exists() else []
    if checkpoint_id not in index:
        index.append(checkpoint_id)
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


def update_runs(object_dir: Path, checkpoint: str = None, data_path: str = None, name: str = None):
    runs_path = object_dir / "runs.json"
    runs = json.load(open(runs_path)) if runs_path.exists() else {}
    if name: runs["name"] = name
    if checkpoint: runs["checkpoint"] = checkpoint
    if data_path: runs["data_path"] = data_path
    with open(runs_path, 'w') as f:
        json.dump(runs, f, indent=2)


def update_model_tags(output_dir: Path, checkpoint_id: str, object_id: str, run_name: str, tag: str):
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
# MAIN
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--output", type=str, default="results")
parser.add_argument("--sequence", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--tags", type=str, nargs="*", default=None)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--frames", type=int, nargs="*", default=None)


def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frame_indices = args.frames if args.frames else FRAME_INDICES
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
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

    # Check for annotations
    annotations_dir = args.image_folder.rstrip("/") + "_annotations"
    has_annotations = os.path.isdir(annotations_dir)

    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if not has_annotations:
        raise RuntimeError(f"Annotations not found at {annotations_dir}. This hybrid version requires annotations.")

    print("=" * 60)
    print("HYBRID MODE:")
    print("  - Image loading: SIMPLE (like test_co3d.py)")
    print("  - GT processing: TRAINING PIPELINE (load_co3d_batch)")
    print("=" * 60)

    # Step 1: Load GT via training pipeline
    print("\n[1] Loading GT via training pipeline...")
    batch_train = load_co3d_batch(args.image_folder, annotations_dir, split="train",
                                   sequence=args.sequence, frame_indices=frame_indices, img_size=args.img_size)

    # Get the actual frame order from batch
    ids = batch_train["ids"][0].tolist()
    frame_labels = [f"t{frame_indices[i]}" for i in ids]
    print(f"Batch order (ids): {ids} -> labels: {frame_labels}")

    # Get GT cameras
    gt_pose_enc, gt_extrinsic, gt_depths, gt_point_masks_np, gt_world_points = get_gt_cameras(batch_train, frame_indices=None)
    print(f"GT pose_enc shape: {gt_pose_enc.shape}")
    print(f"GT depths shape: {gt_depths.shape if gt_depths is not None else None}")
    print(f"GT world_points shape: {gt_world_points.shape if gt_world_points is not None else None}")

    # Step 2: Get image paths in the same order as batch
    print("\n[2] Loading images via SIMPLE preprocessing...")
    # Find original images
    image_dir = Path(args.image_folder)
    all_images = sorted(glob.glob(str(image_dir / "**/images/*.png"), recursive=True))
    all_images += sorted(glob.glob(str(image_dir / "**/images/*.jpg"), recursive=True))

    # Get images at frame_indices, then reorder according to batch ids
    selected_images = [all_images[i] for i in frame_indices if i < len(all_images)]
    # Reorder to match batch order
    reordered_images = [selected_images[i] for i in ids]

    print(f"Loading {len(reordered_images)} images with simple preprocessing")
    images = load_images_simple(reordered_images, target_size=args.img_size).to(device)
    print(f"Image tensor shape: {images.shape}")

    # Step 3: Run inference
    print("\n[3] Running inference...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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

    # Step 4: Export
    object_id, checkpoint_id, data_id = get_object_id(checkpoint_path, args.image_folder)
    frames_str = "_".join(str(i) for i in frame_indices)
    run_name = f"f{frames_str}"
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
        original_images=reordered_images,
        gt_pose_enc=gt_pose_enc,
    )

    # Update index
    data_name = Path(args.image_folder).name
    update_index(output_dir, checkpoint_id)
    tag = args.tags[0] if args.tags else data_name
    update_runs(output_dir / object_id, checkpoint=checkpoint_path, data_path=args.image_folder, name=data_name)
    update_model_tags(output_dir, checkpoint_id, object_id, run_name, tag)

    print(f"\nExport complete! Data saved to: {run_dir}")


if __name__ == "__main__":
    main()
