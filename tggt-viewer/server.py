#!/usr/bin/env python3
"""
Simple server for the VGGT viewer that can run inference on demand.
Caches results so repeated requests are fast.

Usage:
    python server.py [--port PORT] [--checkpoint PATH]

Example:
    python server.py --port 8080 --checkpoint /path/to/checkpoint.pt
"""

import os
import sys
import json
import hashlib
import subprocess
import threading
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Configuration (can be overridden via command line)
PORT = 8080
RESULTS_DIR = Path(__file__).parent / "results"
EXPORT_SCRIPT = Path(__file__).parent / "export_to_viewer.py"

# Default checkpoint (can be overridden via command line or API)
DEFAULT_CHECKPOINT = "/home/chuanruo/vggt_train/training/logs/exp197/ckpts/checkpoint_190.pt"

# Track running jobs
running_jobs = {}
job_lock = threading.Lock()


def get_run_id(object_id, experiment, epoch, frames, split):
    """Generate unique ID for a specific run configuration."""
    key = f"{object_id}_{experiment}_ep{epoch}_{frames}_{split}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def get_run_path(object_id, experiment, epoch, frames=None, split="train"):
    """Get the path where results would be stored."""
    # Base path uses experiment and epoch
    base_name = f"{experiment}_epoch{epoch:03d}"

    # If custom frames or split, add suffix
    if frames or split != "train":
        suffix_parts = []
        if frames:
            suffix_parts.append(f"f{frames.replace(',', '_')}")
        if split and split != "train":
            suffix_parts.append(split)
        if suffix_parts:
            base_name += "_" + "_".join(suffix_parts)

    return RESULTS_DIR / object_id / base_name


def check_results_exist(object_id, experiment, epoch, frames=None, split="train"):
    """Check if results already exist for this configuration."""
    run_path = get_run_path(object_id, experiment, epoch, frames, split)
    # Check for key files
    required = ["cameras.json", "metrics.json", "points_pointmap.ply"]
    return all((run_path / f).exists() for f in required)


def run_export(data_path, checkpoint, epoch, frames=None, split="train", output_dir=None):
    """Run the export script in the vggt conda environment."""
    # Build the python command
    python_cmd = f'python {EXPORT_SCRIPT} --data "{data_path}" --checkpoint "{checkpoint}" --epoch {epoch} --device cpu'

    if output_dir:
        python_cmd += f' --output "{output_dir}"'

    if frames:
        python_cmd += f' --frames "{frames}"'

    if split == "val":
        python_cmd += " --val-only"
    elif split == "all":
        python_cmd += " --val"

    # Wrap in conda activate
    cmd = [
        "bash", "-c",
        f'source /home/chuanruo/anaconda3/etc/profile.d/conda.sh && conda activate vggt && {python_cmd}'
    ]

    print(f"Running: {python_cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Export failed: {result.stderr}")
        return False, result.stderr

    print(f"Export complete: {result.stdout[-500:]}")
    return True, result.stdout


class ViewerHandler(SimpleHTTPRequestHandler):
    """HTTP handler with API endpoints for running inference."""

    def do_GET(self):
        parsed = urlparse(self.path)

        # API endpoint to check/run inference
        if parsed.path == "/api/run":
            self.handle_run_request(parse_qs(parsed.query))
            return

        # API endpoint to check job status
        if parsed.path == "/api/status":
            self.handle_status_request(parse_qs(parsed.query))
            return

        # API endpoint to list available data
        if parsed.path == "/api/list_data":
            self.handle_list_data()
            return

        # Serve static files
        super().do_GET()

    def handle_run_request(self, params):
        """Handle request to run inference."""
        print(f"[API] /run called with params: {params}")
        try:
            # Required params
            data_path = params.get("data", [None])[0]
            if not data_path:
                self.send_error_json(400, "Missing 'data' parameter")
                return

            # Optional params
            checkpoint = params.get("checkpoint", [DEFAULT_CHECKPOINT])[0]
            epoch = int(params.get("epoch", [0])[0])
            frames = params.get("frames", [None])[0]
            split = params.get("split", ["train"])[0]

            print(f"[API] data={data_path}, epoch={epoch}, split={split}")

            # Determine object_id and experiment from data_path
            data_path = Path(data_path)
            experiment = data_path.name
            object_id = data_path.parent.parent.name
            print(f"[API] object_id={object_id}, experiment={experiment}")

            # Check if results exist
            run_path = get_run_path(object_id, experiment, epoch, frames, split)

            if check_results_exist(object_id, experiment, epoch, frames, split):
                # Results exist, return path
                self.send_json({
                    "status": "ready",
                    "path": str(run_path.relative_to(RESULTS_DIR.parent)),
                    "message": "Results already exist"
                })
                return

            # Check if job is already running
            job_id = get_run_id(object_id, experiment, epoch, frames or "", split)

            with job_lock:
                if job_id in running_jobs:
                    self.send_json({
                        "status": "running",
                        "job_id": job_id,
                        "message": "Export already in progress"
                    })
                    return

                # Start new job
                running_jobs[job_id] = {
                    "status": "running",
                    "object_id": object_id,
                    "experiment": experiment,
                    "epoch": epoch,
                    "frames": frames,
                    "split": split
                }

            # Run export in background thread
            def run_job():
                try:
                    success, output = run_export(
                        str(data_path),
                        checkpoint,
                        epoch,
                        frames,
                        split,
                        str(RESULTS_DIR)
                    )
                    with job_lock:
                        running_jobs[job_id]["status"] = "complete" if success else "failed"
                        running_jobs[job_id]["output"] = output[-1000:]
                except Exception as e:
                    with job_lock:
                        running_jobs[job_id]["status"] = "failed"
                        running_jobs[job_id]["error"] = str(e)

            thread = threading.Thread(target=run_job)
            thread.start()

            self.send_json({
                "status": "started",
                "job_id": job_id,
                "message": "Export started"
            })

        except Exception as e:
            self.send_error_json(500, str(e))

    def handle_status_request(self, params):
        """Check status of a running job."""
        job_id = params.get("job_id", [None])[0]

        if not job_id:
            self.send_error_json(400, "Missing 'job_id' parameter")
            return

        with job_lock:
            if job_id not in running_jobs:
                self.send_json({"status": "unknown", "message": "Job not found"})
                return

            job = running_jobs[job_id].copy()

        # If complete, include the result path
        if job["status"] == "complete":
            run_path = get_run_path(
                job["object_id"],
                job["experiment"],
                job["epoch"],
                job.get("frames"),
                job.get("split", "train")
            )
            job["path"] = str(run_path.relative_to(RESULTS_DIR.parent))

        self.send_json(job)

    def handle_list_data(self):
        """List available data directories."""
        data_base = Path("/home/chuanruo/TGGT/out")

        if not data_base.exists():
            self.send_json({"directories": []})
            return

        dirs = []
        for d in sorted(data_base.iterdir()):
            if d.is_dir():
                subsets = d / "subsets"
                if subsets.exists():
                    for s in sorted(subsets.iterdir()):
                        if s.is_dir() and not s.name.endswith("_annotations"):
                            dirs.append({
                                "path": str(s),
                                "object": d.name,
                                "subset": s.name
                            })

        self.send_json({"directories": dirs[:100]})  # Limit to first 100

    def send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_error_json(self, code, message):
        """Send error as JSON."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())

    def log_message(self, format, *args):
        """Custom log format - only show API requests."""
        try:
            msg = str(args[0]) if args else ""
            if "/api/" in msg:
                print(f"[API] {msg}")
        except:
            pass  # Silently ignore logging errors


def main():
    global PORT, DEFAULT_CHECKPOINT

    parser = argparse.ArgumentParser(description="VGGT Viewer Server with on-demand inference")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on (default: 8080)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Default checkpoint for inference")
    args = parser.parse_args()

    PORT = args.port
    DEFAULT_CHECKPOINT = args.checkpoint

    os.chdir(Path(__file__).parent)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           VGGT Viewer Server                             ║
╠══════════════════════════════════════════════════════════╣
║  Server running at: http://localhost:{PORT}               ║
║  Results directory: {RESULTS_DIR}
║  Checkpoint: {Path(DEFAULT_CHECKPOINT).name}
║                                                          ║
║  API Endpoints:                                          ║
║    /api/run?data=PATH&epoch=N&frames=X&split=Y           ║
║    /api/status?job_id=ID                                 ║
║    /api/list_data                                        ║
╚══════════════════════════════════════════════════════════╝
    """)

    server = HTTPServer(("", PORT), ViewerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
