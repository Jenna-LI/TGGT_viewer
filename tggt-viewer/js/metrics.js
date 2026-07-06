// Simplified Metrics Page
let dataByCheckpoint = {};  // checkpointId -> { objectId -> { id, name, metricsByTag } }
let allCheckpoints = [];
let selectedCheckpoints = [];
let allTags = new Set();
let tagComparisonChart = null;
let objectChart = null;

const TAG_COLORS = [
    '#3b82f6', '#f97316', '#10b981', '#8b5cf6', '#ec4899',
    '#06b6d4', '#f59e0b', '#ef4444', '#84cc16', '#6366f1'
];

const METRICS = [
    // Original pose encoding metrics
    { key: 'pose.mean_loss_T', label: 'Translation Enc', color: '#3b82f6' },
    { key: 'pose.mean_loss_R', label: 'Rotation Enc', color: '#8b5cf6' },
    { key: 'pose.mean_loss_FL', label: 'Focal Length', color: '#ec4899' },
    // Camera pose metrics (RRA, RTA, AUC)
    { key: 'camera_pose.rra_mean', label: 'RRA (deg)', color: '#06b6d4' },
    { key: 'camera_pose.rta_mean', label: 'RTA (deg)', color: '#14b8a6' },
    { key: 'camera_pose.auc_30', label: 'AUC@30', color: '#0ea5e9' },
    { key: 'camera_pose.auc_10', label: 'AUC@10', color: '#0284c7' },
    // Depth metrics
    { key: 'depth.mean_depth_mae', label: 'Depth MAE', color: '#10b981' },
    // Pointmap metrics
    { key: 'pointmap.mean_pointmap_mae', label: 'PointMap MAE', color: '#f59e0b' },
    // Chamfer metrics
    { key: 'chamfer.chamfer_accuracy', label: 'Chamfer Acc', color: '#f97316' },
    { key: 'chamfer.chamfer_completeness', label: 'Chamfer Comp', color: '#fb923c' },
    { key: 'chamfer.chamfer_overall', label: 'Chamfer Overall', color: '#ea580c' }
];

// Aggregate metrics across multiple runs, weighted by num_frames
function aggregateMetrics(metricsArray) {
    if (metricsArray.length === 0) return null;
    if (metricsArray.length === 1) return metricsArray[0];

    let totalFrames = 0;
    let totalPairs = 0;
    let totalPoints = 0;
    const weighted = {
        pose: { mean_loss_T: 0, mean_loss_R: 0, mean_loss_FL: 0 },
        camera_pose: { rra_mean: 0, rta_mean: 0, pose_error_mean: 0, auc_3: 0, auc_5: 0, auc_10: 0, auc_30: 0 },
        depth: { mean_depth_mae: 0, mean_depth_rmse: 0 },
        pointmap: { mean_pointmap_mae: 0, mean_pointmap_rmse: 0 },
        chamfer: { chamfer_accuracy: 0, chamfer_completeness: 0, chamfer_overall: 0 }
    };

    for (const m of metricsArray) {
        const n = m.num_frames || 1;
        const np = m.camera_pose?.num_pairs || 0;
        const npts = m.chamfer?.num_pred_points || 0;
        totalFrames += n;
        totalPairs += np;
        totalPoints += npts;

        if (m.pose) {
            weighted.pose.mean_loss_T += (m.pose.mean_loss_T || 0) * n;
            weighted.pose.mean_loss_R += (m.pose.mean_loss_R || 0) * n;
            weighted.pose.mean_loss_FL += (m.pose.mean_loss_FL || 0) * n;
        }
        if (m.camera_pose) {
            weighted.camera_pose.rra_mean += (m.camera_pose.rra_mean || 0) * np;
            weighted.camera_pose.rta_mean += (m.camera_pose.rta_mean || 0) * np;
            weighted.camera_pose.pose_error_mean += (m.camera_pose.pose_error_mean || 0) * np;
            weighted.camera_pose.auc_3 += (m.camera_pose.auc_3 || 0) * np;
            weighted.camera_pose.auc_5 += (m.camera_pose.auc_5 || 0) * np;
            weighted.camera_pose.auc_10 += (m.camera_pose.auc_10 || 0) * np;
            weighted.camera_pose.auc_30 += (m.camera_pose.auc_30 || 0) * np;
        }
        if (m.depth) {
            weighted.depth.mean_depth_mae += (m.depth.mean_depth_mae || 0) * n;
            weighted.depth.mean_depth_rmse += (m.depth.mean_depth_rmse || 0) * n;
        }
        if (m.pointmap) {
            weighted.pointmap.mean_pointmap_mae += (m.pointmap.mean_pointmap_mae || 0) * n;
            weighted.pointmap.mean_pointmap_rmse += (m.pointmap.mean_pointmap_rmse || 0) * n;
        }
        if (m.chamfer) {
            weighted.chamfer.chamfer_accuracy += (m.chamfer.chamfer_accuracy || 0) * npts;
            weighted.chamfer.chamfer_completeness += (m.chamfer.chamfer_completeness || 0) * npts;
            weighted.chamfer.chamfer_overall += (m.chamfer.chamfer_overall || 0) * npts;
        }
    }

    if (totalFrames > 0) {
        weighted.pose.mean_loss_T /= totalFrames;
        weighted.pose.mean_loss_R /= totalFrames;
        weighted.pose.mean_loss_FL /= totalFrames;
        weighted.depth.mean_depth_mae /= totalFrames;
        weighted.depth.mean_depth_rmse /= totalFrames;
        weighted.pointmap.mean_pointmap_mae /= totalFrames;
        weighted.pointmap.mean_pointmap_rmse /= totalFrames;
    }
    if (totalPairs > 0) {
        weighted.camera_pose.rra_mean /= totalPairs;
        weighted.camera_pose.rta_mean /= totalPairs;
        weighted.camera_pose.pose_error_mean /= totalPairs;
        weighted.camera_pose.auc_3 /= totalPairs;
        weighted.camera_pose.auc_5 /= totalPairs;
        weighted.camera_pose.auc_10 /= totalPairs;
        weighted.camera_pose.auc_30 /= totalPairs;
    }
    if (totalPoints > 0) {
        weighted.chamfer.chamfer_accuracy /= totalPoints;
        weighted.chamfer.chamfer_completeness /= totalPoints;
        weighted.chamfer.chamfer_overall /= totalPoints;
    }

    return {
        num_frames: totalFrames,
        pose: weighted.pose,
        camera_pose: totalPairs > 0 ? weighted.camera_pose : null,
        depth: weighted.depth,
        pointmap: weighted.pointmap,
        chamfer: totalPoints > 0 ? weighted.chamfer : null
    };
}

function getM(obj, path) {
    if (!obj) return null;
    const [a, b] = path.split('.');
    return obj[a]?.[b];
}

function avg(arr) {
    const valid = arr.filter(x => x !== null && x !== undefined && !isNaN(x));
    return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
}

async function init() {
    document.getElementById('loading-state').textContent = 'Loading metrics...';

    // Load checkpoint IDs from index.json
    allCheckpoints = await (await fetch('results/index.json?t=' + Date.now())).json();

    for (const checkpointId of allCheckpoints) {
        dataByCheckpoint[checkpointId] = {};

        // Load tags.json for this checkpoint
        let checkpointTags = { tags: {} };
        try {
            const r = await fetch(`results/${checkpointId}/tags.json?t=${Date.now()}`);
            if (r.ok) checkpointTags = await r.json();
        } catch (e) { continue; }

        // Derive unique objects from run paths
        const objectSet = new Set();
        for (const runPaths of Object.values(checkpointTags.tags || {})) {
            for (const runPath of runPaths) {
                const dataId = runPath.split('/')[0];
                objectSet.add(dataId);
            }
        }

        // Process each object
        for (const dataId of objectSet) {
            const id = `${checkpointId}/${dataId}`;

            // Load runs.json for display name
            let runsConfig = null;
            try {
                const r = await fetch(`results/${id}/runs.json?t=${Date.now()}`);
                if (r.ok) runsConfig = await r.json();
            } catch (e) {}

            // Find tags and load metrics
            const metricsByTag = {};
            for (const [tag, runPaths] of Object.entries(checkpointTags.tags || {})) {
                const paths = runPaths.filter(p => p.startsWith(dataId + '/'));
                if (paths.length === 0) continue;

                allTags.add(tag);

                // Load and aggregate metrics for this tag
                const allRunMetrics = [];
                for (const path of paths) {
                    try {
                        const r = await fetch(`results/${checkpointId}/${path}/metrics.json`);
                        if (r.ok) allRunMetrics.push(await r.json());
                    } catch (e) {}
                }

                if (allRunMetrics.length > 0) {
                    metricsByTag[tag] = aggregateMetrics(allRunMetrics);
                }
            }

            if (Object.keys(metricsByTag).length > 0) {
                let objName = dataId;
                if (dataId.includes('_run_')) objName = dataId.split('_run_')[0];

                dataByCheckpoint[checkpointId][id] = {
                    id,
                    name: runsConfig?.name || objName,
                    metricsByTag
                };
            }
        }
    }

    selectedCheckpoints = [...allCheckpoints];

    document.getElementById('loading-state').style.display = 'none';
    document.getElementById('metrics-content').style.display = 'block';

    setupUI();
    render();
}

function getCurrentData() {
    // Merge data from all selected checkpoints
    const merged = {};
    for (const cp of selectedCheckpoints) {
        const cpData = dataByCheckpoint[cp] || {};
        for (const [id, obj] of Object.entries(cpData)) {
            merged[id] = obj;
        }
    }
    return merged;
}

function setupUI() {
    // Checkpoint checkboxes
    const cpContainer = document.getElementById('checkpoint-checkboxes');
    cpContainer.innerHTML = allCheckpoints.map((cp, i) => {
        const color = TAG_COLORS[i % TAG_COLORS.length];
        return `<label class="chip checked">
            <input type="checkbox" class="checkpoint-cb" value="${cp}" data-color="${color}" checked>
            <span class="color-dot" style="background:${color}"></span>
            ${cp}
        </label>`;
    }).join('');
    cpContainer.querySelectorAll('input').forEach(cb => {
        cb.onchange = () => {
            cb.parentElement.classList.toggle('checked', cb.checked);
            selectedCheckpoints = [...document.querySelectorAll('.checkpoint-cb:checked')].map(c => c.value);
            updateCheckpointStats();
            updateObjectCheckboxes();
            render();
        };
    });
    updateCheckpointStats();

    // Tag checkboxes
    const tagContainer = document.getElementById('tag-checkboxes');
    const sortedTags = [...allTags].sort();
    tagContainer.innerHTML = sortedTags.map((tag, i) => {
        const color = TAG_COLORS[i % TAG_COLORS.length];
        return `<label class="chip checked">
            <input type="checkbox" class="tag-cb" value="${tag}" data-color="${color}" checked>
            <span class="color-dot" style="background:${color}"></span>
            ${tag}
        </label>`;
    }).join('');
    tagContainer.querySelectorAll('input').forEach(cb => {
        cb.onchange = () => { cb.parentElement.classList.toggle('checked', cb.checked); render(); };
    });

    // Metric checkboxes - default to original metrics
    const defaultMetrics = ['pose.mean_loss_T', 'pose.mean_loss_R', 'depth.mean_depth_mae', 'pointmap.mean_pointmap_mae'];
    const metricContainer = document.getElementById('metric-checkboxes');
    metricContainer.innerHTML = METRICS.map((m) => {
        const isDefault = defaultMetrics.includes(m.key);
        return `
        <label class="chip ${isDefault ? 'checked' : ''}">
            <input type="checkbox" class="metric-cb" value="${m.key}" ${isDefault ? 'checked' : ''}>
            <span class="color-dot" style="background:${m.color}"></span>
            ${m.label}
        </label>`;
    }).join('');
    metricContainer.querySelectorAll('input').forEach(cb => {
        cb.onchange = () => { cb.parentElement.classList.toggle('checked', cb.checked); render(); };
    });

    // Object checkboxes
    updateObjectCheckboxes();

    // Object tag checkboxes
    updateObjectTagCheckboxes();
}

function updateCheckpointStats() {
    const data = getCurrentData();
    const numObjects = Object.keys(data).length;
    let totalFrames = 0;
    for (const obj of Object.values(data)) {
        for (const metrics of Object.values(obj.metricsByTag)) {
            totalFrames += metrics.num_frames || 0;
        }
    }
    const cpCount = selectedCheckpoints.length;
    document.getElementById('checkpoint-stats').textContent =
        `${cpCount} checkpoint${cpCount !== 1 ? 's' : ''}, ${numObjects} objects, ${totalFrames} frames`;
}

function updateObjectCheckboxes() {
    const container = document.getElementById('object-checkboxes');
    const data = getCurrentData();
    const objects = Object.values(data).sort((a, b) => a.name.localeCompare(b.name));

    container.innerHTML = objects.map((obj, i) => {
        const color = TAG_COLORS[i % TAG_COLORS.length];
        return `<label class="chip checked">
            <input type="checkbox" class="object-cb" value="${obj.id}" data-color="${color}" checked>
            <span class="color-dot" style="background:${color}"></span>
            ${obj.name}
        </label>`;
    }).join('');

    container.querySelectorAll('input').forEach(cb => {
        cb.onchange = () => {
            cb.parentElement.classList.toggle('checked', cb.checked);
            renderObjectChart();
        };
    });
}

function updateObjectTagCheckboxes() {
    const container = document.getElementById('object-tag-checkboxes');
    const sortedTags = [...allTags].sort();

    container.innerHTML = sortedTags.map((tag, i) => {
        const color = TAG_COLORS[i % TAG_COLORS.length];
        return `<label class="chip checked">
            <input type="checkbox" class="object-tag-cb" value="${tag}" data-color="${color}" checked>
            <span class="color-dot" style="background:${color}"></span>
            ${tag}
        </label>`;
    }).join('');

    container.querySelectorAll('input').forEach(cb => {
        cb.onchange = () => {
            cb.parentElement.classList.toggle('checked', cb.checked);
            renderObjectChart();
        };
    });
}

function render() {
    renderTagComparisonChart();
    renderObjectChart();
}

function renderTagComparisonChart() {
    if (tagComparisonChart) tagComparisonChart.destroy();

    const selectedTags = [...document.querySelectorAll('.tag-cb:checked')].map(cb => cb.value);
    const selectedMetrics = [...document.querySelectorAll('.metric-cb:checked')].map(cb => cb.value);

    if (selectedTags.length === 0 || selectedMetrics.length === 0) return;

    const allObjects = Object.values(getCurrentData());
    const metricsToShow = METRICS.filter(m => selectedMetrics.includes(m.key));
    const labels = metricsToShow.map(m => m.label);

    const datasets = [];
    const statsData = [];

    selectedTags.forEach((tag, tagIdx) => {
        const objsWithTag = allObjects.filter(d => d.metricsByTag[tag]);
        if (objsWithTag.length === 0) return;

        const tagCb = document.querySelector(`.tag-cb[value="${tag}"]`);
        const color = tagCb?.dataset.color || TAG_COLORS[tagIdx % TAG_COLORS.length];

        const avgData = metricsToShow.map(m => avg(objsWithTag.map(d => getM(d.metricsByTag[tag], m.key))));
        const totalFrames = objsWithTag.reduce((sum, d) => sum + (d.metricsByTag[tag]?.num_frames || 0), 0);

        datasets.push({
            label: `${tag} (${objsWithTag.length} obj, ${totalFrames} frames)`,
            data: avgData,
            backgroundColor: color + 'cc',
            borderColor: color,
            borderWidth: 1
        });

        statsData.push({ tag, objects: objsWithTag.length, frames: totalFrames });
    });

    if (!datasets.length) return;

    tagComparisonChart = new Chart(document.getElementById('tag-comparison-chart'), {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: { y: { beginAtZero: true } }
        }
    });

    // Stats row
    const statsContainer = document.getElementById('tag-stats');
    statsContainer.innerHTML = statsData.map(s => `
        <div class="stat-card">
            <div class="value">${s.frames}</div>
            <div class="label">${s.tag} frames (${s.objects} objects)</div>
        </div>
    `).join('');
}

function renderObjectChart() {
    if (objectChart) objectChart.destroy();

    const selectedObjectIds = [...document.querySelectorAll('.object-cb:checked')].map(cb => cb.value);
    const selectedTags = [...document.querySelectorAll('.object-tag-cb:checked')].map(cb => cb.value);
    const selectedMetrics = [...document.querySelectorAll('.metric-cb:checked')].map(cb => cb.value);

    if (selectedObjectIds.length === 0 || selectedTags.length === 0 || selectedMetrics.length === 0) return;

    const data = getCurrentData();
    const metricsToShow = METRICS.filter(m => selectedMetrics.includes(m.key));

    // Create labels: "ObjectName (tag)"
    const labels = [];
    const dataPoints = [];  // { objName, tag, metrics }

    for (const objId of selectedObjectIds) {
        const obj = data[objId];
        if (!obj) continue;

        for (const tag of selectedTags) {
            if (obj.metricsByTag[tag]) {
                labels.push(`${obj.name} (${tag})`);
                dataPoints.push({ obj, tag, metrics: obj.metricsByTag[tag] });
            }
        }
    }

    if (dataPoints.length === 0) return;

    // One dataset per metric
    const datasets = metricsToShow.map((m, i) => ({
        label: m.label,
        data: dataPoints.map(dp => getM(dp.metrics, m.key)),
        backgroundColor: m.color + 'cc',
        borderColor: m.color,
        borderWidth: 1
    }));

    objectChart = new Chart(document.getElementById('object-chart'), {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { ticks: { maxRotation: 45, minRotation: 45 } },
                y: { beginAtZero: true }
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', init);
