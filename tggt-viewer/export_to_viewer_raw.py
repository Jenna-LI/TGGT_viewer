#!/usr/bin/env python3
"""
Export VGGT inference results to static web viewer format.
RAW VERSION: Uses raw GT annotations like test_co3d.py (no training pipeline normalization).

Usage:
    python export_to_viewer_raw.py --co3d_dir /path/to/co3d --co3d_anno_dir /path/to/anno --category apple --output results
"""

import os
import sys
import glob
import json
import gzip
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image
from itertools import combinations

# Try to import scipy's cKDTree, fall back to pure numpy implementation
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except (ImportError, ValueError):
    HAS_SCIPY = False

sys.path.insert(0, "/home/chuanruo/vggt")

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.load_fn import load_and_preprocess_images

# =============================================================================
# DEFAULTS
# =============================================================================
CHECKPOINT = "/home/chuanruo/vggt_train/vggt_checkpoints/model.pt"
IMG_SIZE = 518
USE_LORA = False
# =============================================================================


def convert_pt3d_RT_to_opencv(Rot, Trans):
    """
    Convert Point3D extrinsic matrices to OpenCV convention.
    Same as test_co3d.py.
    """
    rot_pt3d = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    return extri_opencv


# =============================================================================
# CAMERA POSE METRICS (RRA, RTA, AUC) - Same as test_co3d.py
# =============================================================================

def closed_form_inverse_se3_np(se3):
    """
    Compute inverse of SE3 matrix. For [R|t], inverse is [R^T | -R^T @ t].
    """
    R = se3[..., :3, :3]
    t = se3[..., :3, 3:4]

    R_inv = np.swapaxes(R, -1, -2)
    t_inv = -R_inv @ t

    shape = se3.shape[:-2]
    result = np.zeros(shape + (4, 4), dtype=se3.dtype)
    result[..., :3, :3] = R_inv
    result[..., :3, 3:4] = t_inv
    result[..., 3, 3] = 1.0
    return result


def mat_to_quat(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    def _sqrt_positive_part(x):
        return np.sqrt(np.maximum(x, 0))

    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    q_abs = _sqrt_positive_part(np.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], axis=-1)) / 2.0

    quat_by_case = np.stack([
        np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
        np.stack([m21 - m12, q_abs[..., 1] ** 2, m01 + m10, m02 + m20], axis=-1),
        np.stack([m02 - m20, m01 + m10, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
        np.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], axis=-1),
    ], axis=-2)

    case_idx = np.argmax(q_abs, axis=-1)
    quat_unnorm = np.take_along_axis(quat_by_case, case_idx[..., None, None], axis=-2).squeeze(-2)

    quat_norm = np.linalg.norm(quat_unnorm, axis=-1, keepdims=True)
    return quat_unnorm / np.maximum(quat_norm, 1e-12)


def rotation_angle(rot_gt, rot_pred, eps=1e-15):
    """Compute rotation angle error using quaternions (same as test_co3d.py)."""
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    dot_product = np.sum(q_pred * q_gt, axis=-1)
    loss_q = np.clip(1 - dot_product ** 2, eps, None)
    err_q = np.arccos(np.clip(1 - 2 * loss_q, -1.0, 1.0))

    return np.degrees(err_q)


def translation_angle(t_gt, t_pred, eps=1e-15):
    """Compute translation angle error (same as test_co3d.py)."""
    t_pred_norm = np.linalg.norm(t_pred, axis=-1, keepdims=True)
    t_pred_unit = t_pred / np.maximum(t_pred_norm, eps)

    t_gt_norm = np.linalg.norm(t_gt, axis=-1, keepdims=True)
    t_gt_unit = t_gt / np.maximum(t_gt_norm, eps)

    dot_product = np.sum(t_pred_unit * t_gt_unit, axis=-1)
    loss_t = np.clip(1.0 - dot_product ** 2, eps, None)
    err_t = np.arccos(np.clip(np.sqrt(1 - loss_t), -1.0, 1.0))

    return np.degrees(err_t)


def compute_auc(errors, thresholds):
    """Compute Area Under the Curve for accuracy-vs-threshold."""
    if len(errors) == 0:
        return {t: 0.0 for t in thresholds}

    errors = np.array(errors)
    auc_results = {}

    for max_thresh in thresholds:
        sweep = np.linspace(0, max_thresh, 100)
        accuracies = np.array([np.mean(errors < tau) for tau in sweep])
        dx = sweep[1] - sweep[0]
        auc = np.sum((accuracies[:-1] + accuracies[1:]) * dx / 2) / max_thresh
        auc_results[max_thresh] = float(auc)

    return auc_results


def compute_camera_pose_metrics(pred_extrinsics, gt_extrinsics):
    """
    Compute camera pose metrics (RRA, RTA, AUC) over all pairs of views.
    Uses proper SE3 relative pose computation (same as test_co3d.py).
    """
    N = len(pred_extrinsics)
    if N < 2:
        return None

    # Convert to 4x4 SE3 matrices
    pred_se3 = np.zeros((N, 4, 4), dtype=np.float64)
    pred_se3[:, :3, :] = pred_extrinsics
    pred_se3[:, 3, 3] = 1.0

    gt_se3 = np.zeros((N, 4, 4), dtype=np.float64)
    gt_se3[:, :3, :] = gt_extrinsics
    gt_se3[:, 3, 3] = 1.0

    rra_errors = []
    rta_errors = []
    pair_errors = []

    for i, j in combinations(range(N), 2):
        relative_pose_gt = gt_se3[i] @ closed_form_inverse_se3_np(gt_se3[j:j+1])[0]
        relative_pose_pred = pred_se3[i] @ closed_form_inverse_se3_np(pred_se3[j:j+1])[0]

        rra = rotation_angle(
            relative_pose_gt[:3, :3],
            relative_pose_pred[:3, :3]
        )
        rta = translation_angle(
            relative_pose_gt[:3, 3],
            relative_pose_pred[:3, 3]
        )

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


def export_raw_data(
    pred_dict,
    output_dir: Path,
    gt_extrinsic=None,
    frame_labels=None,
    original_images=None,
):
    """Export raw inference data for on-demand projection in viewer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    images = pred_dict["images"]
    world_points = pred_dict["world_points"]
    world_points_conf = pred_dict["world_points_conf"]
    depth = pred_dict["depth"]
    depth_conf = pred_dict["depth_conf"]
    extrinsics = pred_dict["extrinsic"]
    intrinsics = pred_dict["intrinsic"]

    S, C, H, W = images.shape
    labels = frame_labels if frame_labels is not None else [f"t{i}" for i in range(S)]

    # Compute scene center
    all_points = world_points.reshape(-1, 3)
    scene_center = np.mean(all_points, axis=0)

    # Use numpy version of closed_form_inverse
    ext_4x4 = np.zeros((S, 4, 4), dtype=extrinsics.dtype)
    ext_4x4[:, :3, :] = extrinsics
    ext_4x4[:, 3, 3] = 1.0
    cam_to_world_mat = closed_form_inverse_se3_np(ext_4x4)
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

    for i in range(S):
        img = images[i].transpose(1, 2, 0)
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(output_dir / "images" / f"view_{i:03d}.png")

        depth_i = depth[i].squeeze(-1).astype(np.float32)
        depth_i.tofile(output_dir / "depths" / f"view_{i:03d}.bin")

        pointmap_i = (world_points[i] - scene_center).astype(np.float32)
        pointmap_i.tofile(output_dir / "pointmaps" / f"view_{i:03d}.bin")

        conf_i = world_points_conf[i].astype(np.float32)
        conf_i.tofile(output_dir / "confidence" / f"view_{i:03d}.bin")

        depth_conf_i = depth_conf[i].astype(np.float32)
        depth_conf_i.tofile(output_dir / "depth_conf" / f"view_{i:03d}.bin")

    print(f"  Wrote {S} frames (images, depths, pointmaps, confidence)")

    # Build cameras data
    pred_cameras = []
    for i in range(S):
        fy = 1.1 * H
        fov_rad = 2 * np.arctan2(H / 2, fy)
        fov_deg = float(np.degrees(fov_rad))

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

    # GT cameras if available
    gt_cameras = []
    if gt_extrinsic is not None:
        gt_ext_4x4 = np.zeros((len(gt_extrinsic), 4, 4), dtype=np.float64)
        gt_ext_4x4[:, :3, :] = gt_extrinsic
        gt_ext_4x4[:, 3, 3] = 1.0
        gt_cam2world = closed_form_inverse_se3_np(gt_ext_4x4)[:, :3, :]
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

    cameras_data = {
        "num_frames": S,
        "height": H,
        "width": W,
        "scene_center": scene_center.tolist(),
        "depth_range": [float(depth.min()), float(depth.max())],
        "conf_range": [float(world_points_conf.min()), float(world_points_conf.max())],
        "pred_cameras": pred_cameras,
        "gt_cameras": gt_cameras if gt_cameras else None,
        "has_gt_pointmaps": False,
        "has_gt_depths": False,
        "has_gt_masks": False,
    }

    with open(output_dir / "cameras.json", 'w') as f:
        json.dump(cameras_data, f, indent=2)
    print(f"  Wrote cameras.json")

    # Compute metrics
    metrics = {"num_frames": S}

    if gt_extrinsic is not None and len(gt_extrinsic) >= 2:
        camera_pose_metrics = compute_camera_pose_metrics(extrinsics, gt_extrinsic)
        if camera_pose_metrics is not None:
            metrics["camera_pose"] = camera_pose_metrics

    metrics["conf_min"] = float(world_points_conf.min())
    metrics["conf_max"] = float(world_points_conf.max())
    metrics["mean_depth"] = float(depth.mean())
    metrics["mode"] = "raw_gt"  # Mark this as using raw GT

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Wrote metrics.json")


parser = argparse.ArgumentParser()
parser.add_argument("--co3d_dir", type=str, required=True, help="Path to CO3D dataset root")
parser.add_argument("--co3d_anno_dir", type=str, required=True, help="Path to CO3D raw annotations (.jgz files)")
parser.add_argument("--category", type=str, required=True, help="CO3D category (e.g., apple)")
parser.add_argument("--sequence", type=str, default=None, help="Specific sequence name (random if not specified)")
parser.add_argument("--output", type=str, default="results_raw")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to sample")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--frame_ids", type=int, nargs="*", default=None, help="Specific frame IDs to use (overrides random sampling)")


def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
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

    # Set random seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load raw annotations (same as test_co3d.py)
    annotation_file = os.path.join(args.co3d_anno_dir, f"{args.category}_test.jgz")
    print(f"Loading annotations from {annotation_file}...")

    with gzip.open(annotation_file, "r") as fin:
        annotation = json.loads(fin.read())

    # Get sequence
    seq_names = sorted(list(annotation.keys()))
    if args.sequence:
        if args.sequence not in seq_names:
            raise ValueError(f"Sequence '{args.sequence}' not found. Available: {seq_names[:10]}...")
        seq_name = args.sequence
    else:
        seq_name = random.choice(seq_names)

    seq_data = annotation[seq_name]
    print(f"Using sequence: {seq_name} ({len(seq_data)} frames)")

    # Build metadata with converted extrinsics (same as test_co3d.py)
    metadata = []
    for data in seq_data:
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            continue
        extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({
            "filepath": data["filepath"],
            "extri": extri_opencv,
        })

    if len(metadata) < args.num_frames:
        raise ValueError(f"Sequence has only {len(metadata)} valid frames, need {args.num_frames}")

    # Sample frames (same as test_co3d.py)
    if args.frame_ids is not None:
        ids = np.array(args.frame_ids)
        print(f"Using specified frame IDs: {ids}")
    else:
        ids = np.random.choice(len(metadata), args.num_frames, replace=False)
        print(f"Randomly sampled frame IDs: {sorted(ids)}")

    image_names = [os.path.join(args.co3d_dir, metadata[i]["filepath"]) for i in ids]
    gt_extri = np.stack([metadata[i]["extri"] for i in ids], axis=0)

    # Load and preprocess images (SAME AS test_co3d.py - NO training pipeline!)
    print(f"Loading {len(image_names)} images via load_and_preprocess_images...")
    images = load_and_preprocess_images(image_names).to(device)

    print(f"Image tensor shape: {images.shape}")

    # Run inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in predictions:
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    # Create frame labels
    frame_labels = [f"id{i}" for i in ids]

    # Create run directory
    ids_str = "_".join(str(i) for i in sorted(ids))
    run_dir = output_dir / f"{args.category}_{seq_name}" / f"frames_{ids_str}"

    print(f"\n{'='*60}")
    print(f"Exporting to {run_dir}...")
    print(f"MODE: RAW GT (same as test_co3d.py)")
    print(f"{'='*60}")

    export_raw_data(
        pred_dict=predictions,
        output_dir=run_dir,
        gt_extrinsic=gt_extri,
        frame_labels=frame_labels,
        original_images=image_names,
    )

    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS (RAW GT mode - same as test_co3d.py):")
    print(f"{'='*60}")

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    if "camera_pose" in metrics:
        cp = metrics["camera_pose"]
        print(f"  RRA mean:  {cp['rra_mean']:.4f}°")
        print(f"  RTA mean:  {cp['rta_mean']:.4f}°")
        print(f"  AUC@3:     {cp['auc_3']:.4f}")
        print(f"  AUC@5:     {cp['auc_5']:.4f}")
        print(f"  AUC@10:    {cp['auc_10']:.4f}")
        print(f"  AUC@30:    {cp['auc_30']:.4f}")

    print(f"\nExport complete! Data saved to: {run_dir}")


if __name__ == "__main__":
    main()
