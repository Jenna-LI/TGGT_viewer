#!/bin/bash

source /home/chuanruo/anaconda3/etc/profile.d/conda.sh
conda activate vggt

CHECKPOINT="/home/chuanruo/vggt_train/vggt_checkpoints/model.pt"
FRAMES="1 3 5 7 9 17 23"
TAG="co3d_data"

CO3D_DIR="/home/chuanruo/co3d_data"
CO3D_ANNO_DIR="/home/chuanruo/co3d_data/annotations_converted"

CATEGORIES=(
    "apple"
    # Add more categories here
)

for CATEGORY in "${CATEGORIES[@]}"; do
    echo "Processing: $CATEGORY"
    python export_to_viewer.py \
        --co3d_dir "$CO3D_DIR" \
        --co3d_anno_dir "$CO3D_ANNO_DIR" \
        --category "$CATEGORY" \
        --checkpoint "$CHECKPOINT" \
        --tags "$TAG" \
        --frames $FRAMES \
        --img_size 518 \
        --output results
    echo "---"
done

echo "Done!"
