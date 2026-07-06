// 3D Viewer - Raw data format (loads binary pointmaps/depths/confidence on demand)
// Falls back to PLY format for old exports
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const COLOR_PRED = 0x1e40af;      // Dark blue for predicted
const COLOR_GT = 0x0e7490;        // Dark cyan for ground truth
const COLOR_DESELECTED = 0x999999; // Grey for deselected frames

const CAMERA_STATE_KEY = 'vggt_viewer_camera_state';

class ScenePanel {
    constructor(canvasId, onFrustumSelect) {
        this.canvas = document.getElementById(canvasId);
        this.onFrustumSelect = onFrustumSelect;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf5f5f5);

        const aspect = this.canvas.clientWidth / this.canvas.clientHeight || 1;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.01, 100);
        this.camera.position.set(2, 2, 2);

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.5));
        this.scene.add(new THREE.AxesHelper(0.2));

        this.pointCloud = null;
        this.gtPointCloud = null;
        this.predHelpers = [];
        this.gtHelpers = [];
        this.errorLines = [];
        this.cameraData = [];
        this.basePath = null;

        // Raw data storage (per-frame)
        this.frameData = [];  // Array of {pointmap, depth, conf, depthConf, colors, H, W}
        this.gtDepths = [];   // GT depths for gt_depth modes
        this.metadata = null;  // cameras.json data
        this.pointConfThreshold = 2.5;  // For pointmap mode
        this.depthConfThreshold = 2.5;  // For depth unprojection modes
        this.currentPointSize = 0.01;
        this.selectedFrames = null;  // Set of selected frame indices, null = all
        this.pointCloudMode = 'pointmap';  // 'pointmap', 'pred_depth_pred_pose', 'pred_depth_gt_pose', 'gt_depth_pred_pose', 'gt_depth_gt_pose'

        // Raycaster for click detection
        this.raycaster = new THREE.Raycaster();
        this.raycaster.params.Line.threshold = 0.05;
        this.canvas.addEventListener('dblclick', (e) => this.onDoubleClick(e));

        this.resize();
    }

    onDoubleClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1
        );
        this.raycaster.setFromCamera(mouse, this.camera);

        const allHelpers = [...this.predHelpers, ...this.gtHelpers];
        for (const helper of allHelpers) {
            const intersects = this.raycaster.intersectObject(helper, true);
            if (intersects.length > 0) {
                const idx = helper.userData.idx;
                const camData = this.cameraData[idx];
                if (camData) {
                    this.snapToView(camData);
                    if (this.onFrustumSelect) {
                        this.onFrustumSelect(idx, camData);
                    }
                }
                return;
            }
        }
    }

    snapToView(camData) {
        const pos = new THREE.Vector3(...camData.position);
        this.camera.position.copy(pos);

        if (camData.matrix) {
            const m = camData.matrix;
            // Use same forward convention as frustum visualization (Z axis direction)
            const forward = new THREE.Vector3(m[0][2], m[1][2], m[2][2]);
            const target = pos.clone().add(forward);
            this.controls.target.copy(target);
            this.camera.lookAt(target);
        }
        this.controls.update();
    }

    resize() {
        const parent = this.canvas.parentElement;
        const rect = parent.getBoundingClientRect();
        const w = Math.floor(rect.width);
        const h = Math.floor(rect.height);
        if (w > 0 && h > 0) {
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(w, h);
        }
    }

    render() {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    clear() {
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
            this.pointCloud = null;
        }
        if (this.gtPointCloud) {
            this.scene.remove(this.gtPointCloud);
            this.gtPointCloud.geometry.dispose();
            this.gtPointCloud.material.dispose();
            this.gtPointCloud = null;
        }
        [...this.predHelpers, ...this.gtHelpers].forEach(h => this.scene.remove(h));
        this.errorLines.forEach(l => { this.scene.remove(l); l.geometry.dispose(); l.material.dispose(); });
        this.predHelpers = [];
        this.gtHelpers = [];
        this.errorLines = [];
        this.basePath = null;
        this.frameData = [];
        this.gtDepths = [];
        this.metadata = null;
        this.selectedFrames = null;
    }

    /**
     * Load raw data from new format, with fallback to old PLY format.
     * @param {string} path - Base path to run directory
     */
    async loadRawData(path) {
        this.basePath = path;

        // Load cameras.json (contains all metadata)
        const resp = await fetch(`${path}/cameras.json?t=${Date.now()}`);
        if (!resp.ok) throw new Error('Failed to load cameras.json');
        this.metadata = await resp.json();

        // Check if this is new format (has height/width) or old PLY format
        if (!this.metadata.height || !this.metadata.width) {
            console.log('Old PLY format detected, using fallback loader');
            return this.loadOldFormat(path);
        }

        const numFrames = this.metadata.num_frames;
        const H = this.metadata.height;
        const W = this.metadata.width;

        console.log(`Loading ${numFrames} frames (${H}x${W})`);

        // Load all frame data in parallel
        this.frameData = [];
        const loadPromises = [];

        for (let i = 0; i < numFrames; i++) {
            const idx = String(i).padStart(3, '0');
            loadPromises.push(this.loadFrameData(path, idx, H, W));
        }

        this.frameData = await Promise.all(loadPromises);

        // Set initial confidence thresholds from metadata
        if (this.metadata.conf_range) {
            // Default to showing ~75% of points
            const defaultThresh = this.metadata.conf_range[0] +
                (this.metadata.conf_range[1] - this.metadata.conf_range[0]) * 0.25;
            this.pointConfThreshold = defaultThresh;
            this.depthConfThreshold = defaultThresh;
        }

        console.log(`Loaded ${this.frameData.length} frames, point conf: ${this.pointConfThreshold.toFixed(2)}, depth conf: ${this.depthConfThreshold.toFixed(2)}`);

        // Build initial point cloud
        this.updatePointCloud();

        return this.metadata;
    }

    /**
     * Load data for a single frame.
     */
    async loadFrameData(basePath, idx, H, W) {
        const [pointmapResp, confResp, depthResp, depthConfResp, imgResp] = await Promise.all([
            fetch(`${basePath}/pointmaps/view_${idx}.bin`),
            fetch(`${basePath}/confidence/view_${idx}.bin`),
            fetch(`${basePath}/depths/view_${idx}.bin`),
            fetch(`${basePath}/depth_conf/view_${idx}.bin`),
            fetch(`${basePath}/images/view_${idx}.png`)
        ]);

        // Pointmap (H x W x 3 float32)
        const pointmapBuffer = await pointmapResp.arrayBuffer();
        const pointmap = new Float32Array(pointmapBuffer);

        // Point confidence (H x W float32)
        const confBuffer = await confResp.arrayBuffer();
        const conf = new Float32Array(confBuffer);

        // Depth (H x W float32)
        const depthBuffer = await depthResp.arrayBuffer();
        const depth = new Float32Array(depthBuffer);

        // Depth confidence (H x W float32)
        const depthConfBuffer = await depthConfResp.arrayBuffer();
        const depthConf = new Float32Array(depthConfBuffer);

        // Colors from image
        const imgBlob = await imgResp.blob();
        const imgBitmap = await createImageBitmap(imgBlob);
        const canvas = new OffscreenCanvas(W, H);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imgBitmap, 0, 0);
        const imgData = ctx.getImageData(0, 0, W, H);
        const colors = new Uint8Array(H * W * 3);
        for (let i = 0; i < H * W; i++) {
            colors[i * 3] = imgData.data[i * 4];
            colors[i * 3 + 1] = imgData.data[i * 4 + 1];
            colors[i * 3 + 2] = imgData.data[i * 4 + 2];
        }

        return { pointmap, depth, conf, depthConf, colors, H, W };
    }

    /**
     * Fallback loader for old PLY format exports.
     */
    async loadOldFormat(path) {
        console.log('Loading old PLY format from:', path);
        const loader = new PLYLoader();

        // Load main point cloud
        const plyPath = `${path}/points_all.ply?t=${Date.now()}`;
        const geometry = await new Promise((resolve, reject) => {
            loader.load(plyPath, resolve, undefined, reject);
        });

        geometry.computeBoundingBox();

        const material = new THREE.PointsMaterial({
            vertexColors: true,
            size: this.currentPointSize,
            sizeAttenuation: true
        });

        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);

        // Try to load GT points
        try {
            const gtPlyPath = `${path}/points_gt.ply?t=${Date.now()}`;
            const gtGeometry = await new Promise((resolve, reject) => {
                loader.load(gtPlyPath, resolve, undefined, reject);
            });

            const gtMaterial = new THREE.PointsMaterial({
                vertexColors: true,
                size: this.currentPointSize,
                sizeAttenuation: true
            });

            this.gtPointCloud = new THREE.Points(gtGeometry, gtMaterial);
            this.gtPointCloud.visible = false;
            this.scene.add(this.gtPointCloud);
        } catch (e) {
            console.log('No GT points available');
        }

        // Return metadata-like object for compatibility
        return {
            pred_cameras: this.metadata.pred_cameras || [],
            gt_cameras: this.metadata.gt_cameras,
            num_frames: (this.metadata.pred_cameras || []).length,
            conf_range: null,
            depth_range: null,
            is_old_format: true
        };
    }

    /**
     * Load GT pointmaps and GT depths if available.
     */
    async loadGTPointmaps() {
        if (!this.basePath) return;

        const numFrames = this.metadata.num_frames;
        const gtData = [];
        this.gtDepths = [];

        // Load GT pointmaps
        if (this.metadata?.has_gt_pointmaps) {
            for (let i = 0; i < numFrames; i++) {
                const idx = String(i).padStart(3, '0');
                try {
                    const resp = await fetch(`${this.basePath}/gt_pointmaps/view_${idx}.bin`);
                    if (resp.ok) {
                        const buffer = await resp.arrayBuffer();
                        gtData.push(new Float32Array(buffer));
                    }
                } catch (e) {
                    console.warn(`Failed to load GT pointmap ${idx}`);
                }
            }
        }

        // Load GT depths
        if (this.metadata?.has_gt_depths) {
            for (let i = 0; i < numFrames; i++) {
                const idx = String(i).padStart(3, '0');
                try {
                    const resp = await fetch(`${this.basePath}/gt_depths/view_${idx}.bin`);
                    if (resp.ok) {
                        const buffer = await resp.arrayBuffer();
                        this.gtDepths.push(new Float32Array(buffer));
                    }
                } catch (e) {
                    console.warn(`Failed to load GT depth ${idx}`);
                }
            }
        }

        if (gtData.length === 0) return;

        // Build GT point cloud (cyan)
        const allPoints = [];
        const allColors = [];

        for (const pointmap of gtData) {
            for (let i = 0; i < pointmap.length / 3; i++) {
                allPoints.push(pointmap[i * 3], pointmap[i * 3 + 1], pointmap[i * 3 + 2]);
                allColors.push(0, 200/255, 255/255);  // Cyan
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(allPoints, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(allColors, 3));

        const material = new THREE.PointsMaterial({
            vertexColors: true,
            size: this.currentPointSize,
            sizeAttenuation: true
        });

        this.gtPointCloud = new THREE.Points(geometry, material);
        this.gtPointCloud.visible = false;
        this.scene.add(this.gtPointCloud);
    }

    /**
     * Set point confidence threshold (for pointmap mode).
     */
    setPointConfThreshold(threshold) {
        this.pointConfThreshold = threshold;
        this.updatePointCloud();
    }

    /**
     * Set depth confidence threshold (for depth unprojection modes).
     */
    setDepthConfThreshold(threshold) {
        this.depthConfThreshold = threshold;
        this.updatePointCloud();
    }

    /**
     * Set confidence threshold (legacy - sets both).
     */
    setConfidenceThreshold(threshold) {
        this.pointConfThreshold = threshold;
        this.depthConfThreshold = threshold;
        this.updatePointCloud();
    }

    /**
     * Set selected frames for filtering.
     */
    setSelectedFrames(frameIndices) {
        this.selectedFrames = frameIndices;
        this.updatePointCloud();
        this.updateFrustumColors();
    }

    /**
     * Update frustum colors based on selected frames.
     * Deselected frames get grey frustums.
     */
    updateFrustumColors() {
        const isSelected = (idx) => {
            if (this.selectedFrames === null) return true; // All selected
            return this.selectedFrames.has(idx);
        };

        // Update predicted camera frustums
        this.predHelpers.forEach(helper => {
            const idx = helper.userData.idx;
            const color = isSelected(idx) ? COLOR_PRED : COLOR_DESELECTED;
            helper.traverse(child => {
                if (child.material) {
                    child.material.color.setHex(color);
                }
            });
        });

        // Update GT camera frustums
        this.gtHelpers.forEach(helper => {
            const idx = helper.userData.idx;
            const color = isSelected(idx) ? COLOR_GT : COLOR_DESELECTED;
            helper.traverse(child => {
                if (child.material) {
                    child.material.color.setHex(color);
                }
            });
        });

        // Update error lines
        this.errorLines.forEach(line => {
            const idx = line.userData.idx;
            const color = isSelected(idx) ? 0xff0000 : COLOR_DESELECTED;
            if (line.material) {
                line.material.color.setHex(color);
            }
        });
    }

    /**
     * Set point cloud rendering mode.
     */
    setPointCloudMode(mode) {
        this.pointCloudMode = mode;
        this.updatePointCloud();
    }

    /**
     * Unproject depth map to 3D points using intrinsics and camera pose.
     */
    unprojectDepth(depth, intrinsic, c2w, H, W, sceneCenter) {
        const points = [];
        const fx = intrinsic[0][0];
        const fy = intrinsic[1][1];
        const cx = intrinsic[0][2];
        const cy = intrinsic[1][2];

        for (let v = 0; v < H; v++) {
            for (let u = 0; u < W; u++) {
                const idx = v * W + u;
                const d = depth[idx];
                if (d <= 0 || !isFinite(d)) {
                    points.push(NaN, NaN, NaN);
                    continue;
                }

                // Camera space point
                const x_cam = (u - cx) * d / fx;
                const y_cam = (v - cy) * d / fy;
                const z_cam = d;

                // Transform to world space using c2w matrix (3x4)
                const x_world = c2w[0][0] * x_cam + c2w[0][1] * y_cam + c2w[0][2] * z_cam + c2w[0][3];
                const y_world = c2w[1][0] * x_cam + c2w[1][1] * y_cam + c2w[1][2] * z_cam + c2w[1][3];
                const z_world = c2w[2][0] * x_cam + c2w[2][1] * y_cam + c2w[2][2] * z_cam + c2w[2][3];

                points.push(x_world, y_world, z_world);
            }
        }
        return new Float32Array(points);
    }

    /**
     * Rebuild point cloud from raw data with current filters.
     */
    updatePointCloud() {
        if (this.frameData.length === 0) return;

        const positions = [];
        const colors = [];
        const mode = this.pointCloudMode;
        const usePointmap = (mode === 'pointmap');
        // Use both thresholds simultaneously (like viewer2.js/eval_viser.py)
        const pointThreshold = this.pointConfThreshold;
        const depthThreshold = this.depthConfThreshold;

        for (let frameIdx = 0; frameIdx < this.frameData.length; frameIdx++) {
            // Frame filter
            if (this.selectedFrames !== null && !this.selectedFrames.has(frameIdx)) {
                continue;
            }

            const frame = this.frameData[frameIdx];
            const numPixels = frame.conf.length;
            const H = frame.H;
            const W = frame.W;

            // Get points based on mode
            let points;
            if (mode === 'pointmap') {
                points = frame.pointmap;
            } else {
                // Depth unprojection modes
                const useGtDepth = mode.startsWith('gt_depth');
                const useGtPose = mode.endsWith('gt_pose');

                // Get camera data
                const predCam = this.metadata.pred_cameras?.[frameIdx];
                const gtCam = this.metadata.gt_cameras?.[frameIdx];

                // Determine which depth and pose to use
                const depth = useGtDepth ? (this.gtDepths?.[frameIdx] || frame.depth) : frame.depth;
                const cam = useGtPose ? (gtCam || predCam) : predCam;

                if (!cam || !cam.intrinsic || !cam.matrix) {
                    points = frame.pointmap; // Fallback to pointmap
                } else {
                    points = this.unprojectDepth(depth, cam.intrinsic, cam.matrix, H, W);
                }
            }

            for (let i = 0; i < numPixels; i++) {
                // Apply BOTH confidence thresholds (like viewer2.js/eval_viser.py)
                // Point confidence filter (always applied)
                if (frame.conf[i] < pointThreshold || frame.conf[i] <= 0.1) continue;
                // Depth confidence filter (applied if available)
                if (frame.depthConf && (frame.depthConf[i] < depthThreshold || frame.depthConf[i] <= 0.1)) continue;

                // Get position
                const x = points[i * 3];
                const y = points[i * 3 + 1];
                const z = points[i * 3 + 2];

                // Skip invalid points
                if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;

                positions.push(x, y, z);
                colors.push(
                    frame.colors[i * 3] / 255,
                    frame.colors[i * 3 + 1] / 255,
                    frame.colors[i * 3 + 2] / 255
                );
            }
        }

        console.log(`Point cloud: ${positions.length / 3} points (pointConf >= ${pointThreshold.toFixed(2)}, depthConf >= ${depthThreshold.toFixed(2)}, frames: ${this.selectedFrames ? this.selectedFrames.size : 'all'})`);

        // Remove old point cloud
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }

        // Create new geometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.computeBoundingBox();

        const material = new THREE.PointsMaterial({
            vertexColors: true,
            size: this.currentPointSize,
            sizeAttenuation: true
        });

        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);
    }

    setGTPointsVisible(v) {
        if (this.gtPointCloud) this.gtPointCloud.visible = v;
    }

    setPointSize(s) {
        this.currentPointSize = s * 0.01;
        if (this.pointCloud?.material) this.pointCloud.material.size = this.currentPointSize;
        if (this.gtPointCloud?.material) this.gtPointCloud.material.size = this.currentPointSize;
    }

    addCameras(data, indices) {
        const pred = data.pred_cameras || data;
        const gt = data.gt_cameras;
        const show = indices || pred.map((_, i) => i);
        this.cameraData = pred;

        for (const i of show) {
            if (pred[i]) {
                const h = this._makeHelper(pred[i], COLOR_PRED);
                if (h) { h.userData.idx = i; this.scene.add(h); this.predHelpers.push(h); }
            }
            if (gt && gt[i]) {
                const h = this._makeHelper(gt[i], COLOR_GT);
                if (h) { h.userData.idx = i; this.scene.add(h); this.gtHelpers.push(h); }
            }
            if (pred[i] && gt && gt[i]) {
                const p1 = new THREE.Vector3(...pred[i].position);
                const p2 = new THREE.Vector3(...gt[i].position);
                const geo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
                const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xff0000 }));
                line.userData.idx = i;
                this.scene.add(line);
                this.errorLines.push(line);
            }
        }
    }

    _makeHelper(cam, color) {
        // Create a simple frustum/pyramid visualization
        const pos = cam.position;
        const size = 0.08;  // Size of the frustum base
        const depth = 0.12; // Depth of the frustum

        // Get camera orientation from matrix or look_at
        let forward, right, up;
        if (cam.matrix) {
            const m = cam.matrix;
            // Extract axes from the 3x4 matrix (columns are right, up, -forward in OpenGL)
            right = new THREE.Vector3(m[0][0], m[1][0], m[2][0]).normalize();
            up = new THREE.Vector3(m[0][1], m[1][1], m[2][1]).normalize();
            forward = new THREE.Vector3(m[0][2], m[1][2], m[2][2]).normalize(); // Z axis direction
        } else if (cam.look_at) {
            const camPos = new THREE.Vector3(pos[0], pos[1], pos[2]);
            const target = new THREE.Vector3(cam.look_at[0], cam.look_at[1], cam.look_at[2]);
            forward = target.clone().sub(camPos).normalize();
            up = new THREE.Vector3(0, 1, 0);
            right = new THREE.Vector3().crossVectors(forward, up).normalize();
            up = new THREE.Vector3().crossVectors(right, forward).normalize();
        } else {
            forward = new THREE.Vector3(0, 0, -1);
            up = new THREE.Vector3(0, 1, 0);
            right = new THREE.Vector3(1, 0, 0);
        }

        const origin = new THREE.Vector3(pos[0], pos[1], pos[2]);

        // Compute frustum corners at depth
        const aspect = cam.aspect || 1;
        const halfW = size * aspect;
        const halfH = size;

        const center = origin.clone().add(forward.clone().multiplyScalar(depth));
        const tl = center.clone().add(up.clone().multiplyScalar(halfH)).add(right.clone().multiplyScalar(-halfW));
        const tr = center.clone().add(up.clone().multiplyScalar(halfH)).add(right.clone().multiplyScalar(halfW));
        const bl = center.clone().add(up.clone().multiplyScalar(-halfH)).add(right.clone().multiplyScalar(-halfW));
        const br = center.clone().add(up.clone().multiplyScalar(-halfH)).add(right.clone().multiplyScalar(halfW));

        // Create line segments for the frustum
        const points = [
            // Lines from origin to corners
            origin, tl,
            origin, tr,
            origin, bl,
            origin, br,
            // Rectangle at far plane
            tl, tr,
            tr, br,
            br, bl,
            bl, tl
        ];

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ color: color, linewidth: 2 });
        const frustum = new THREE.LineSegments(geometry, material);

        return frustum;
    }

    setPredVisible(v) { this.predHelpers.forEach(h => h.visible = v); }
    setGtVisible(v) { this.gtHelpers.forEach(h => h.visible = v); }
    setLinesVisible(v) { this.errorLines.forEach(l => l.visible = v); }

    frameCamera() {
        if (!this.pointCloud) return;
        const box = new THREE.Box3().setFromObject(this.pointCloud);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const dist = Math.max(size.x, size.y, size.z) * 2.5 || 3;

        this.camera.position.set(center.x + dist, center.y + dist * 0.5, center.z + dist);
        this.controls.target.set(center.x, center.y, center.z);
        this.camera.lookAt(center);
        this.controls.update();
    }

    resetView() {
        this.resize();
        this.frameCamera();
    }
}

class Viewer {
    constructor() {
        this.checkpointIds = [];
        this.checkpointTags = {};
        this.objectIds = [];
        this.runsConfigs = {};
        this.leftPanel = null;
        this.rightPanel = null;
        this.splitView = false;
        this.syncCameras = false;

        this.left = { objectId: null, tag: null, runIndex: 0, split: 'train', frames: null, metrics: null, cameras: null, fullMetrics: null };
        this.right = { objectId: null, tag: null, runIndex: 0, split: 'train', frames: null, metrics: null, cameras: null, fullMetrics: null };

        this.lastCameraState = { pos: null, target: null };
        this.currentRunPath = { left: null, right: null };

        this.init();
    }

    async init() {
        // Load checkpoint IDs from index.json
        this.checkpointIds = await fetch('results/index.json').then(r => r.json());

        // Load tags.json for each checkpoint and derive objects
        this.objectIds = [];
        for (const checkpointId of this.checkpointIds) {
            const tags = await this.loadCheckpointTags(checkpointId);
            // Derive unique object IDs from run paths in tags
            const objectSet = new Set();
            for (const runPaths of Object.values(tags.tags || {})) {
                for (const runPath of runPaths) {
                    // runPath is like "baboon_run_.../f1_3_5_7"
                    // objectId is checkpoint + first part of runPath
                    const dataId = runPath.split('/')[0];
                    objectSet.add(`${checkpointId}/${dataId}`);
                }
            }
            this.objectIds.push(...objectSet);
        }

        // Load runs.json for each object (for display names)
        await Promise.all(this.objectIds.map(id => this.loadRunsConfig(id)));

        const params = new URLSearchParams(window.location.search);
        const initObj = params.get('id') || this.objectIds[0];

        this.leftPanel = new ScenePanel('canvas-left', (idx, cam) => this.onFrustumSelect('left', idx, cam));
        this.rightPanel = new ScenePanel('canvas-right', (idx, cam) => this.onFrustumSelect('right', idx, cam));

        this.leftPanel.controls.addEventListener('change', () => this.saveCameraState());

        this.populateObjectSelects();

        // Get tags for initial object from checkpoint's tags.json
        const checkpointId = this.getCheckpointId(initObj);
        const tags = this.checkpointTags[checkpointId];
        const tagNames = this.getTagsForObject(initObj, tags);

        this.left.objectId = initObj;
        this.left.tag = tagNames[0];

        this.right.objectId = initObj;
        this.right.tag = tagNames[0];

        this.updateSelectValues('left');
        this.updateSelectValues('right');

        this.setupEvents();
        await this.loadPanel('left');

        this.animate();
    }

    async loadCheckpointTags(checkpointId) {
        if (!this.checkpointTags[checkpointId]) {
            try {
                this.checkpointTags[checkpointId] = await fetch(`results/${checkpointId}/tags.json`).then(r => r.json());
            } catch (e) {
                this.checkpointTags[checkpointId] = { tags: {} };
            }
        }
        return this.checkpointTags[checkpointId];
    }

    getTagsForObject(objectId, tagsData) {
        // Find which tags have runs for this object
        const checkpointId = this.getCheckpointId(objectId);
        const dataId = objectId.slice(checkpointId.length + 1);
        const tagNames = [];
        for (const [tag, runPaths] of Object.entries(tagsData?.tags || {})) {
            if (runPaths.some(p => p.startsWith(dataId + '/'))) {
                tagNames.push(tag);
            }
        }
        return tagNames;
    }

    async loadRunsConfig(objectId) {
        if (!this.runsConfigs[objectId]) {
            try {
                this.runsConfigs[objectId] = await fetch(`results/${objectId}/runs.json`).then(r => r.json());
            } catch (e) {
                this.runsConfigs[objectId] = { name: objectId.split('/').pop() };
            }
        }
        return this.runsConfigs[objectId];
    }

    getCheckpointId(objectId) {
        // Extract checkpoint_id from objectId
        // objectId: "exp197_ckpts_checkpoint_190/baboon_run_20260613_014414_my_data_125"
        // checkpoint_id is the first path component
        return objectId.split('/')[0];
    }

    saveCameraState() {
        const state = {
            id: this.left.objectId,
            position: this.leftPanel.camera.position.toArray(),
            target: this.leftPanel.controls.target.toArray(),
            zoom: this.leftPanel.camera.zoom
        };
        localStorage.setItem(CAMERA_STATE_KEY, JSON.stringify(state));
    }

    restoreCameraState() {
        try {
            const saved = localStorage.getItem(CAMERA_STATE_KEY);
            if (saved) {
                const state = JSON.parse(saved);
                if (state.id === this.left.objectId) {
                    this.leftPanel.camera.position.fromArray(state.position);
                    this.leftPanel.controls.target.fromArray(state.target);
                    if (state.zoom) this.leftPanel.camera.zoom = state.zoom;
                    this.leftPanel.camera.updateProjectionMatrix();
                    this.leftPanel.controls.update();
                    return true;
                }
            }
        } catch (e) {
            console.warn('Could not restore camera state:', e);
        }
        return false;
    }

    populateObjectSelects() {
        ['left', 'right'].forEach(side => {
            const sel = document.getElementById(`${side}-object-select`);
            sel.innerHTML = this.objectIds.map(id => {
                const name = this.runsConfigs[id]?.name || id;
                return `<option value="${id}">${name}</option>`;
            }).join('');
        });
    }

    async populateTagSelect(side) {
        const state = this[side];
        const checkpointId = this.getCheckpointId(state.objectId);
        const tagsData = this.checkpointTags[checkpointId] || { tags: {} };
        const tagNames = this.getTagsForObject(state.objectId, tagsData);

        const sel = document.getElementById(`${side}-exp-select`);
        sel.innerHTML = tagNames.map(t => `<option value="${t}">${t}</option>`).join('');
        // Hide epoch select since we don't use it anymore
        document.getElementById(`${side}-epoch-select`).style.display = 'none';
        // Populate run select for current tag
        await this.populateRunSelect(side);
    }

    async populateRunSelect(side) {
        const state = this[side];
        const checkpointId = this.getCheckpointId(state.objectId);
        const tagsData = this.checkpointTags[checkpointId] || { tags: {} };
        const dataId = state.objectId.slice(checkpointId.length + 1);  // +1 for "/"

        // Filter runs for this object and tag
        const allRuns = tagsData.tags?.[state.tag] || [];
        const paths = allRuns.filter(p => p.startsWith(dataId + '/'));

        const sel = document.getElementById(`${side}-run-select`);
        sel.innerHTML = paths.map((p, i) => {
            // Extract just the run name from path
            const runName = p.split('/').pop();
            // Extract frames from run name (e.g., "f1_3_5_7_9_17_23" -> "f1,3,5,7,9,17,23")
            const matchF = runName.match(/^f([\d_]+)$/);
            const matchV = runName.match(/^v([\d_]+)$/);
            let label = runName;
            if (matchF) label = `train: ${matchF[1].replace(/_/g, ',')}`;
            else if (matchV) label = `val: ${matchV[1].replace(/_/g, ',')}`;
            return `<option value="${i}">${label}</option>`;
        }).join('');
        sel.value = state.runIndex || 0;

        // Store paths for getRunPath
        state.runPaths = paths;
    }

    updateSelectValues(side) {
        const state = this[side];
        document.getElementById(`${side}-object-select`).value = state.objectId;
        this.populateTagSelect(side).then(() => {
            document.getElementById(`${side}-exp-select`).value = state.tag;
            document.getElementById(`${side}-split-select`).value = state.split;
        });
    }

    setupEvents() {
        // Split view toggle
        document.getElementById('toggle-split-view').addEventListener('change', async e => {
            this.splitView = e.target.checked;
            document.getElementById('panel-right').style.display = this.splitView ? 'flex' : 'none';
            document.getElementById('metrics-right').style.display = this.splitView ? 'block' : 'none';
            document.getElementById('sync-label').style.display = this.splitView ? 'inline-flex' : 'none';

            await new Promise(r => setTimeout(r, 150));
            this.leftPanel.resetView();

            if (this.splitView) {
                await this.loadPanel('right');
                await new Promise(r => setTimeout(r, 50));
                this.rightPanel.resetView();
                this.rightPanel.camera.position.copy(this.leftPanel.camera.position);
                this.rightPanel.controls.target.copy(this.leftPanel.controls.target);
                this.rightPanel.controls.update();
            }
        });

        // Sync cameras toggle
        document.getElementById('toggle-sync-cameras').addEventListener('change', e => {
            this.syncCameras = e.target.checked;
            if (this.syncCameras && this.splitView) {
                this.rightPanel.camera.position.copy(this.leftPanel.camera.position);
                this.rightPanel.controls.target.copy(this.leftPanel.controls.target);
            }
        });

        ['left', 'right'].forEach(side => {
            const panel = side === 'left' ? this.leftPanel : this.rightPanel;

            document.getElementById(`${side}-object-select`).addEventListener('change', async e => {
                this[side].objectId = e.target.value;
                const checkpointId = this.getCheckpointId(e.target.value);
                await this.loadCheckpointTags(checkpointId);
                const tagsData = this.checkpointTags[checkpointId];
                const tagNames = this.getTagsForObject(e.target.value, tagsData);
                this[side].tag = tagNames[0];
                this[side].runIndex = 0;  // Reset to first run
                this.updateSelectValues(side);
                this.loadPanel(side);
            });

            document.getElementById(`${side}-exp-select`).addEventListener('change', async e => {
                this[side].tag = e.target.value;
                this[side].runIndex = 0;  // Reset to first run when tag changes
                await this.populateRunSelect(side);
                this.loadPanel(side);
            });

            document.getElementById(`${side}-split-select`).addEventListener('change', async e => {
                this[side].split = e.target.value;
                await this.loadPanel(side);
            });

            document.getElementById(`${side}-run-select`).addEventListener('change', async e => {
                this[side].runIndex = parseInt(e.target.value);
                await this.loadPanel(side);
            });

            document.getElementById(`${side}-reset-view`).addEventListener('click', () => panel.resetView());
            document.getElementById(`${side}-show-pred`).addEventListener('change', e => panel.setPredVisible(e.target.checked));
            document.getElementById(`${side}-show-gt`).addEventListener('change', e => panel.setGtVisible(e.target.checked));
            document.getElementById(`${side}-show-lines`).addEventListener('change', e => panel.setLinesVisible(e.target.checked));
            document.getElementById(`${side}-point-size`).addEventListener('input', e => panel.setPointSize(parseFloat(e.target.value)));

            // GT points toggle
            document.getElementById(`${side}-show-gt-points`).addEventListener('change', e => {
                panel.setGTPointsVisible(e.target.checked);
            });

            // Point confidence threshold (for pointmap mode)
            document.getElementById(`${side}-point-conf-threshold`).addEventListener('change', e => {
                const threshold = parseFloat(e.target.value);
                if (!isNaN(threshold) && threshold >= 0) {
                    panel.setPointConfThreshold(threshold);
                }
            });

            // Depth confidence threshold (for depth unprojection modes)
            document.getElementById(`${side}-depth-conf-threshold`).addEventListener('change', e => {
                const threshold = parseFloat(e.target.value);
                if (!isNaN(threshold) && threshold >= 0) {
                    panel.setDepthConfThreshold(threshold);
                }
            });

            // Point cloud mode (pointmap vs depth unprojection)
            document.getElementById(`${side}-pointcloud-mode`).addEventListener('change', e => {
                panel.setPointCloudMode(e.target.value);
            });
        });

        // Frame buttons
        document.getElementById('btn-all').addEventListener('click', () => this.setAllFrames(true));
        document.getElementById('btn-none').addEventListener('click', () => this.setAllFrames(false));
        document.getElementById('btn-train').addEventListener('click', () => this.setFramesBySplit('train'));
        document.getElementById('btn-val').addEventListener('click', () => this.setFramesBySplit('val'));

        // Exit view button
        document.getElementById('exit-view-btn').addEventListener('click', () => this.exitSelectedView());

        // View images click to enlarge
        document.getElementById('view-rgb').addEventListener('click', (e) => this.showLightbox(e.target.src, 'RGB'));
        document.getElementById('view-depth').addEventListener('click', (e) => this.showLightbox(e.target.src, 'Depth'));

        window.addEventListener('resize', () => {
            this.leftPanel.resize();
            if (this.splitView) this.rightPanel.resize();
        });

        // Lightbox
        document.querySelector('.lightbox-close').addEventListener('click', () => {
            document.getElementById('lightbox').style.display = 'none';
        });
    }

    getRunPath(state) {
        // Use runPaths populated by populateRunSelect
        const paths = state.runPaths || [];
        const idx = state.runIndex || 0;
        const runPath = paths[idx];
        if (!runPath) return null;
        // runPath is relative to checkpoint, need full path
        const checkpointId = this.getCheckpointId(state.objectId);
        return `results/${checkpointId}/${runPath}`;
    }

    async loadPanel(side) {
        const state = this[side];
        const panel = side === 'left' ? this.leftPanel : this.rightPanel;
        const path = this.getRunPath(state);

        document.getElementById('loading-indicator').style.display = 'block';

        try {
            panel.clear();

            // Load raw data (cameras.json + binary files)
            const metadata = await panel.loadRawData(path);

            this.currentRunPath[side] = path;
            state.cameras = metadata;
            state.metrics = {
                num_frames: metadata.num_frames,
                conf_range: metadata.conf_range,
                depth_range: metadata.depth_range
            };

            // Also load GT pointmaps if available
            await panel.loadGTPointmaps();

            // Update confidence inputs
            const pointConfInput = document.getElementById(`${side}-point-conf-threshold`);
            const depthConfInput = document.getElementById(`${side}-depth-conf-threshold`);
            if (metadata.conf_range) {
                pointConfInput.value = panel.pointConfThreshold.toFixed(2);
                pointConfInput.max = Math.ceil(metadata.conf_range[1]);
                pointConfInput.min = Math.floor(metadata.conf_range[0] * 10) / 10;
            }
            // Depth conf uses same range for now (could be different if we track it separately)
            if (metadata.conf_range) {
                depthConfInput.value = panel.depthConfThreshold.toFixed(2);
                depthConfInput.max = Math.ceil(metadata.conf_range[1]);
                depthConfInput.min = Math.floor(metadata.conf_range[0] * 10) / 10;
            }

            // Restore GT points visibility
            const showGTPoints = document.getElementById(`${side}-show-gt-points`).checked;
            panel.setGTPointsVisible(showGTPoints);

            // Add camera helpers
            panel.addCameras(metadata);

            // Restore or frame camera
            if (side === 'left' && !this.restoreCameraState()) {
                panel.frameCamera();
            } else if (side === 'right') {
                panel.frameCamera();
            }

            // Load full metrics.json for per-frame data
            try {
                const metricsResp = await fetch(`${path}/metrics.json?t=${Date.now()}`);
                if (metricsResp.ok) {
                    state.fullMetrics = await metricsResp.json();
                }
            } catch (e) {
                console.warn('Could not load metrics.json:', e);
            }

            this.updateMetrics(side);
            if (side === 'left') {
                this.updateFrameList(metadata);
                this.updatePerFrameMetrics();
            }
            this.updateSelectionInfo();

        } catch (err) {
            console.error(`Error loading ${side}:`, err);
            alert(`Error loading viewer: ${err.message}\n\nCheck console for details.`);
        }

        document.getElementById('loading-indicator').style.display = 'none';
    }

    updateMetrics(side) {
        const state = this[side];
        const container = document.querySelector(`#metrics-${side} .metrics-values`);
        container.innerHTML = '';

        if (!state.metrics) return;

        const metrics = state.metrics;
        const rows = [
            ['Frames', metrics.num_frames],
            ['Conf Range', metrics.conf_range ? `${metrics.conf_range[0].toFixed(2)} - ${metrics.conf_range[1].toFixed(2)}` : 'N/A'],
            ['Depth Range', metrics.depth_range ? `${metrics.depth_range[0].toFixed(2)} - ${metrics.depth_range[1].toFixed(2)}` : 'N/A']
        ];

        rows.forEach(([k, v]) => {
            const row = document.createElement('div');
            row.className = 'metric-row';
            row.innerHTML = `<span class="metric-name">${k}</span><span class="metric-value">${v}</span>`;
            container.appendChild(row);
        });
    }

    updateSelectionInfo() {
        const info = document.getElementById('selection-info');
        const fmt = s => `${s.objectId} / ${s.tag}`;
        info.innerHTML = this.splitView
            ? `<b>L:</b> ${fmt(this.left)}<br><b>R:</b> ${fmt(this.right)}`
            : fmt(this.left);
    }

    updateFrameList(cameras) {
        const container = document.getElementById('frame-list');
        container.innerHTML = '';

        const cams = cameras.pred_cameras || [];

        cams.forEach((c, i) => {
            const label = document.createElement('label');

            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = true;
            cb.dataset.idx = i;
            cb.addEventListener('change', () => {
                this.updateSelectedFramesFilter();
            });

            label.appendChild(cb);
            label.appendChild(document.createTextNode(c.view_id || `f${i}`));

            if (c.source_image) {
                const pathSpan = document.createElement('span');
                pathSpan.style.cssText = 'font-size:9px;color:#888;margin-left:5px;';
                pathSpan.textContent = c.source_image.split('/').pop();
                label.appendChild(pathSpan);
            }
            container.appendChild(label);
        });
    }

    updatePerFrameMetrics() {
        const headerRow = document.getElementById('per-frame-header');
        const body = document.getElementById('per-frame-body');
        headerRow.innerHTML = '';
        body.innerHTML = '';

        const metrics = this.left.fullMetrics;
        if (!metrics) {
            document.getElementById('per-frame-metrics-section').style.display = 'none';
            return;
        }
        document.getElementById('per-frame-metrics-section').style.display = 'block';

        // Collect available per-frame metric types
        const metricTypes = [];
        if (metrics.pose?.per_frame?.length > 0) {
            metricTypes.push({ key: 'pose', cols: ['loss_T', 'loss_R', 'loss_FL'], labels: ['T', 'R', 'FL'] });
        }
        if (metrics.depth?.per_frame?.length > 0) {
            metricTypes.push({ key: 'depth', cols: ['depth_mae', 'depth_rmse'], labels: ['D_MAE', 'D_RMSE'] });
        }
        if (metrics.pointmap?.per_frame?.length > 0) {
            metricTypes.push({ key: 'pointmap', cols: ['pointmap_mae', 'pointmap_rmse'], labels: ['PM_MAE', 'PM_RMSE'] });
        }

        if (metricTypes.length === 0) {
            document.getElementById('per-frame-metrics-section').style.display = 'none';
            return;
        }

        // Build header
        const thFrame = document.createElement('th');
        thFrame.textContent = 'Frame';
        headerRow.appendChild(thFrame);

        for (const mt of metricTypes) {
            for (const label of mt.labels) {
                const th = document.createElement('th');
                th.textContent = label;
                headerRow.appendChild(th);
            }
        }

        // Get frame count from first available metric type
        const firstType = metricTypes[0];
        const perFrame = metrics[firstType.key].per_frame;

        // Build rows
        for (let i = 0; i < perFrame.length; i++) {
            const tr = document.createElement('tr');

            // Frame ID
            const tdFrame = document.createElement('td');
            tdFrame.className = 'frame-id';
            tdFrame.textContent = perFrame[i].frame || `f${i}`;
            tr.appendChild(tdFrame);

            // Metric values
            for (const mt of metricTypes) {
                const frameData = metrics[mt.key]?.per_frame?.[i] || {};
                for (const col of mt.cols) {
                    const td = document.createElement('td');
                    const val = frameData[col];
                    if (val !== undefined && val !== null) {
                        td.textContent = val.toFixed(4);
                        // Color code based on error magnitude
                        if (col.includes('mae') || col.includes('rmse') || col.includes('loss')) {
                            if (val > 0.1) td.className = 'error-high';
                            else if (val > 0.01) td.className = 'error-medium';
                            else td.className = 'error-low';
                        }
                    } else {
                        td.textContent = '-';
                    }
                    tr.appendChild(td);
                }
            }

            body.appendChild(tr);
        }
    }

    updateSelectedFramesFilter() {
        const checkboxes = document.querySelectorAll('#frame-list input[type="checkbox"]');
        const allChecked = Array.from(checkboxes).every(cb => cb.checked);
        const noneChecked = Array.from(checkboxes).every(cb => !cb.checked);

        let selectedFrames = null;
        if (!allChecked && !noneChecked) {
            selectedFrames = new Set();
            checkboxes.forEach(cb => {
                if (cb.checked) {
                    selectedFrames.add(parseInt(cb.dataset.idx));
                }
            });
        } else if (noneChecked) {
            selectedFrames = new Set();
        }

        [this.leftPanel, this.rightPanel].forEach(p => {
            if (p.frameData.length > 0) {
                p.setSelectedFrames(selectedFrames);
            }
        });
    }

    setAllFrames(checked) {
        document.querySelectorAll('#frame-list input').forEach(cb => {
            cb.checked = checked;
        });
        this.updateSelectedFramesFilter();
    }

    setFramesBySplit(split) {
        // For now, just check/uncheck based on view_id prefix
        document.querySelectorAll('#frame-list label').forEach(label => {
            const cb = label.querySelector('input');
            const text = label.textContent;
            const isVal = text.startsWith('v');
            const isTrain = text.startsWith('t');
            cb.checked = (split === 'val' && isVal) || (split === 'train' && isTrain) || split === 'all';
        });
        this.updateSelectedFramesFilter();
    }

    onFrustumSelect(side, idx, camData) {
        const runPath = this.currentRunPath[side];
        if (!runPath) return;

        const section = document.getElementById('selected-view-section');
        section.style.display = 'block';

        document.getElementById('selected-view-id').textContent = camData.view_id || `Frame ${idx}`;

        const rgbImg = document.getElementById('view-rgb');
        const depthImg = document.getElementById('view-depth');

        const idxStr = String(idx).padStart(3, '0');
        rgbImg.src = `${runPath}/images/view_${idxStr}.png`;
        depthImg.src = `${runPath}/depths/view_${idxStr}.bin`;  // Won't display, but placeholder

        // Show depth as grayscale from confidence (hack for now)
        // In a real implementation, you'd render the depth as an image server-side

        const sourcePathEl = document.getElementById('source-image-path');
        sourcePathEl.textContent = camData.source_image || 'N/A';
        this.currentSourcePath = camData.source_image || '';

        this.selectedView = { side, idx, camData, runPath };
    }

    exitSelectedView() {
        document.getElementById('selected-view-section').style.display = 'none';
        if (this.selectedView) {
            const panel = this.selectedView.side === 'left' ? this.leftPanel : this.rightPanel;
            panel.frameCamera();
        }
        this.selectedView = null;
    }

    showLightbox(src, caption) {
        document.getElementById('lightbox-image').src = src;
        document.getElementById('lightbox-caption').textContent = caption;
        document.getElementById('lightbox').style.display = 'flex';
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.splitView && this.syncCameras) {
            const lCam = this.leftPanel.camera;
            const rCam = this.rightPanel.camera;
            const lCtrl = this.leftPanel.controls;
            const rCtrl = this.rightPanel.controls;

            if (this.lastCameraState.pos) {
                const leftMoved = !lCam.position.equals(this.lastCameraState.pos) ||
                                  !lCtrl.target.equals(this.lastCameraState.target);
                if (leftMoved) {
                    rCam.position.copy(lCam.position);
                    rCtrl.target.copy(lCtrl.target);
                }
            }
            this.lastCameraState.pos = lCam.position.clone();
            this.lastCameraState.target = lCtrl.target.clone();
        }

        this.leftPanel.render();
        if (this.splitView) this.rightPanel.render();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.viewerInstance = new Viewer();
});
