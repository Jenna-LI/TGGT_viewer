// Gallery - discovers objects by scanning checkpoint tags.json files
let objects = [];  // Array of {id, name, tags, ...}
let allTags = new Set();  // All unique tags across all checkpoints

async function loadGallery() {
    const loadingEl = document.getElementById('loading-state');
    const errorEl = document.getElementById('error-state');

    try {
        // Load checkpoint IDs from index.json
        const response = await fetch('results/index.json?t=' + Date.now());
        if (!response.ok) throw new Error('Failed to load index.json');
        const checkpointIds = await response.json();

        objects = [];
        allTags = new Set();

        // For each checkpoint, load tags.json and derive objects
        for (const checkpointId of checkpointIds) {
            let checkpointTags = { tags: {} };
            try {
                const tagsResp = await fetch(`results/${checkpointId}/tags.json?t=${Date.now()}`);
                if (tagsResp.ok) checkpointTags = await tagsResp.json();
            } catch (e) {
                continue;
            }

            // Collect all unique tags
            for (const tag of Object.keys(checkpointTags.tags || {})) {
                allTags.add(tag);
            }

            // Derive unique objects from run paths
            const objectSet = new Set();
            for (const runPaths of Object.values(checkpointTags.tags || {})) {
                for (const runPath of runPaths) {
                    const dataId = runPath.split('/')[0];
                    objectSet.add(dataId);
                }
            }

            // Load runs.json for each object
            for (const dataId of objectSet) {
                const id = `${checkpointId}/${dataId}`;
                let runsConfig = { name: dataId };
                try {
                    const runsResp = await fetch(`results/${id}/runs.json?t=${Date.now()}`);
                    if (runsResp.ok) runsConfig = await runsResp.json();
                } catch (e) {}

                // Find tags for this object
                const tags = [];
                for (const [tag, runPaths] of Object.entries(checkpointTags.tags || {})) {
                    if (runPaths.some(p => p.startsWith(dataId + '/'))) {
                        tags.push(tag);
                    }
                }

                objects.push({ id, tags, ...runsConfig });
            }
        }

        loadingEl.style.display = 'none';
        populateTagFilter();
        renderGallery();
    } catch (err) {
        loadingEl.style.display = 'none';
        errorEl.style.display = 'block';
        errorEl.textContent = 'Error: ' + err.message;
    }
}

function populateTagFilter() {
    const filterEl = document.getElementById('tag-filter');
    if (!filterEl) return;

    // Keep "All" option, remove dynamically added ones
    while (filterEl.options.length > 1) {
        filterEl.remove(1);
    }

    // Add options for each unique tag (sorted alphabetically)
    const sortedTags = Array.from(allTags).sort();
    for (const tag of sortedTags) {
        const option = document.createElement('option');
        option.value = tag;
        option.textContent = tag;
        filterEl.appendChild(option);
    }
}

function renderGallery() {
    const gridEl = document.getElementById('gallery-grid');
    const countEl = document.getElementById('gallery-count');
    const tagFilter = document.getElementById('tag-filter')?.value || 'all';
    const searchTerm = document.getElementById('search-box')?.value.toLowerCase() || '';

    gridEl.innerHTML = '';

    let filtered = objects.filter(obj => {
        // tags is an array of tag names
        const tagNames = obj.tags || [];
        // Tag filter
        if (tagFilter !== 'all' && !tagNames.includes(tagFilter)) return false;
        // Search filter
        const name = obj.name || obj.id;
        if (searchTerm && !name.toLowerCase().includes(searchTerm) && !obj.id.toLowerCase().includes(searchTerm)) {
            return false;
        }
        return true;
    });

    // Sort by name
    filtered.sort((a, b) => (a.name || a.id).localeCompare(b.name || b.id));

    countEl.textContent = filtered.length + ' objects';

    filtered.forEach(obj => {
        const card = document.createElement('div');
        card.className = 'gallery-card';
        card.dataset.id = obj.id;

        // Image container
        const imgContainer = document.createElement('div');
        imgContainer.className = 'card-canvas-container';
        const img = document.createElement('img');
        img.src = `results/${obj.id}/gt_cover.png`;
        img.alt = obj.name || obj.id;
        img.loading = 'lazy';
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        img.onerror = () => {
            img.style.background = '#333';
            img.alt = 'No preview';
        };
        imgContainer.appendChild(img);

        // Label
        const label = document.createElement('div');
        label.className = 'card-label';
        label.textContent = obj.name || obj.id;

        // Tag badges - tags is an array
        const tagNames = obj.tags || [];
        if (tagNames.length > 0) {
            const badgeContainer = document.createElement('div');
            badgeContainer.className = 'card-badges';

            tagNames.forEach(tag => {
                const badge = document.createElement('span');
                badge.className = 'card-badge tag-' + tag;
                badge.textContent = tag.toUpperCase();
                badgeContainer.appendChild(badge);
            });

            card.appendChild(badgeContainer);
        }

        // Open viewer link
        const openLink = document.createElement('a');
        openLink.href = `viewer.html?id=${encodeURIComponent(obj.id)}`;
        openLink.className = 'card-open-link';
        openLink.textContent = 'Open Viewer';

        card.appendChild(imgContainer);
        card.appendChild(label);
        card.appendChild(openLink);
        gridEl.appendChild(card);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    loadGallery();
    document.getElementById('tag-filter')?.addEventListener('change', renderGallery);
    document.getElementById('search-box')?.addEventListener('input', renderGallery);
});
