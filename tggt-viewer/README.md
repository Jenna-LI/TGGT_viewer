# 3D Reconstruction Results Viewer

A static website for browsing 3D reconstruction results from research projects. Each result displays a point cloud reconstructed from multiple camera views, similar to COLMAP/NeRF/VGGT-style pipeline visualizations.

## Features

- **Gallery page**: Grid of thumbnail cards for each reconstructed object
- **3D Viewer page**: Interactive Three.js viewer with:
  - Orbit/zoom camera controls
  - Point cloud rendering with vertex colors
  - Toggle between multiple point cloud sources (predicted, ground truth, etc.)
  - Dual camera frustum display: predicted (blue) and ground truth (cyan)
  - Error lines connecting GT↔Pred camera positions (green=low error, red=high)
  - Experiment/epoch selection to toggle between training runs
  - Depth image thumbnails with frustum highlighting
  - Live metrics readout
- **Metrics page**: Numerical error comparison across experiments and epochs
  - Filterable by object or aggregated across all
  - Comparison table with best-value highlighting
  - Trend chart showing metric improvement over training

## Quick Start

1. Open `index.html` in a browser, or serve the folder with any static file server:
   ```bash
   python3 -m http.server 8000
   # Then visit http://localhost:8000
   ```

2. For GitHub Pages: push this folder to a repo and enable Pages in settings.

## Adding New Results

### New Object

1. Create `results/<new_id>/` with:
   - `gt_cover.png` — cover image for gallery card (ground truth, shared across runs)
   - `runs.json` — experiment/epoch configuration (see schema below)

2. For each (experiment, epoch) combination, create a folder (e.g., `exp01_epoch020/`) containing:
   - `points.ply` — predicted point cloud (ASCII PLY with x, y, z, red, green, blue)
   - `points_gt.ply` — (optional) ground truth point cloud
   - `cameras.json` — camera poses for frustum rendering (see schema below)
   - `metrics.json` — numerical error values for this run
   - `depths/` folder with depth images (e.g., `view_000.png`, `view_001.png`, ...)

3. Add entry to `results/manifest.json`:
   ```json
   { "id": "<new_id>", "name": "<Display Name>" }
   ```

4. Append run rows to `results/metrics_summary.json` (see schema below)

### New Experiment/Epoch for Existing Object

1. Add the new run folder under that object's directory
2. Update `runs.json` to include the new experiment name and/or epoch number
3. Append the new row(s) to `results/metrics_summary.json`

**No rebuild step required** — just refresh the page.

## Data Schemas

### `results/manifest.json`

```json
[
  { "id": "sample_001", "name": "Sample Object 01" },
  { "id": "sample_002", "name": "Sample Object 02" }
]
```

### `results/<id>/runs.json`

```json
{
  "experiments": ["exp01", "exp02"],
  "epochs_by_experiment": {
    "exp01": [5, 20, 50],
    "exp02": [5, 20]
  },
  "path_template": "{experiment}_epoch{epoch:03d}"
}
```

The `path_template` uses `{experiment}` and `{epoch:03d}` (zero-padded to 3 digits) placeholders.

### `<run_folder>/cameras.json`

The cameras file supports two formats:

**Format 1: Separate predicted and GT cameras (recommended)**
```json
{
  "pred_cameras": [
    {
      "view_id": "view_000",
      "position": [0.1, 0.0, 2.5],
      "look_at": [0.0, 0.0, 0.0],
      "up": [0.0, 1.0, 0.0],
      "fov_deg": 50,
      "aspect": 1.333,
      "near": 0.01,
      "far": 0.3,
      "depth_image": "depths/view_000.png"
    }
  ],
  "gt_cameras": [
    {
      "view_id": "view_000",
      "position": [0.0, 0.0, 2.5],
      "look_at": [0.0, 0.0, 0.0],
      "up": [0.0, 1.0, 0.0],
      "fov_deg": 50,
      "aspect": 1.333,
      "near": 0.01,
      "far": 0.3
    }
  ]
}
```

**Format 2: Flat array (backward compatible, pred-only)**
```json
[
  {
    "view_id": "view_000",
    "position": [0.0, 0.0, 2.5],
    "look_at": [0.0, 0.0, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fov_deg": 50,
    "aspect": 1.333,
    "near": 0.01,
    "far": 0.3,
    "depth_image": "depths/view_000.png"
  }
]
```

When both `pred_cameras` and `gt_cameras` are provided, the viewer shows:
- Blue frustums for predicted poses
- Cyan frustums for ground truth poses
- Error lines connecting each GT↔Pred pair (color-coded by error magnitude)

### `<run_folder>/metrics.json`

```json
{
  "chamfer_distance": 0.0061,
  "emd": 0.083,
  "surface_error_pct": 4.2
}
```

Metric keys are discovered dynamically — adding new metrics doesn't require code changes.

### `results/metrics_summary.json`

```json
[
  {
    "object_id": "sample_001",
    "experiment": "exp01",
    "epoch": 5,
    "chamfer_distance": 0.0091,
    "emd": 0.131,
    "surface_error_pct": 9.8
  }
]
```

A denormalized rollup of all per-run metrics. Keep in sync with individual `metrics.json` files.

### `<run_folder>/points.ply`

Standard ASCII PLY format:
```
ply
format ascii 1.0
element vertex <count>
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0.1 0.2 0.3 255 128 64
...
```

## Dependencies

All loaded via CDN (no npm install required):
- [Three.js](https://threejs.org/) v0.160.0 — 3D rendering
- [Chart.js](https://www.chartjs.org/) v4.4.1 — Metrics trend chart

## Notes

- This repo is intentionally separate from model training/research code
- It only receives lightweight exported result folders (point clouds, camera poses, depth images, metrics) copied in after eval runs
- Does not contain or depend on any training code, model weights, or large raw datasets
- Lower metric values are assumed to be better (Chamfer Distance, EMD, surface error) — the comparison table highlights minimum values per row

## Regenerating Sample Data

The included sample data was generated with:
```bash
python3 generate_sample_data.py
```

This creates synthetic torus and sphere point clouds with varying noise levels across epochs to demonstrate the viewer functionality.
