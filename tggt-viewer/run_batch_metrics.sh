#!/bin/bash
cd /home/chuanruo/TGGT_viewer/tggt-viewer
source /home/chuanruo/anaconda3/etc/profile.d/conda.sh
conda activate vggt

CHECKPOINT="/home/chuanruo/vggt_train/training/logs/exp197/ckpts/checkpoint_190.pt"
EPOCH=190
FRAMES="3,5,7,9,11,12,34,45,56,67,77,78,89,90"
DATA_BASE="/home/chuanruo/TGGT/out"

count=0
total=$(ls -d ${DATA_BASE}/*/subsets/my_data_125 2>/dev/null | wc -l)

for data_dir in ${DATA_BASE}/*/subsets/my_data_125; do
    [ -d "$data_dir" ] || continue
    count=$((count + 1))
    name=$(basename $(dirname $(dirname "$data_dir")))
    echo "[$count/$total] $name"
    
    python export_to_viewer.py \
        --data "$data_dir" \
        --checkpoint "$CHECKPOINT" \
        --epoch $EPOCH \
        --frames "$FRAMES" \
        --device cpu 2>&1 | grep -E "(Pose errors|Depth errors|PointMap errors|Error|Done)"
done

echo "Batch complete: $count objects"
