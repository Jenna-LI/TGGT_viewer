#!/bin/bash

source /home/chuanruo/anaconda3/etc/profile.d/conda.sh
conda activate vggt

CHECKPOINT="/home/chuanruo/vggt_train/training/logs/exp201/ckpts/checkpoint_200.pt"
FRAMES="12 10 23 33 44 55 66 77 88 99"
TAG="unseen_object"
IMG_SIZE=224

# Path to multi-category data directory (contains airplane/, almond/, etc.)
DATA_DIR="/home/chuanruo/vggt_train/merged_50obj_unseen/subsets/my_data_125"
ANNO_DIR="/home/chuanruo/vggt_train/merged_50obj_unseen/subsets/my_data_125_annotations"

# Process all categories (omit --category to process all)
# --val-only uses test/val split instead of train
# Provide frame indices after --val-only, or omit for all val frames
python export_to_viewer.py \
    --image_folder "$DATA_DIR" \
    --anno_dir "$ANNO_DIR" \
    --checkpoint "$CHECKPOINT" \
    --tags "$TAG" \
    --img_size $IMG_SIZE \
    --output results

# Or process a single category:
# python export_to_viewer.py \
#     --image_folder "$DATA_DIR" \
#     --category "airplane" \
#     --checkpoint "$CHECKPOINT" \
#     --tags "$TAG" \
#     --frames $FRAMES \
#     --img_size $IMG_SIZE \
#     --output results

echo "Done!"
