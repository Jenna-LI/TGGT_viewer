#!/bin/bash

source /home/chuanruo/anaconda3/etc/profile.d/conda.sh
conda activate vggt

CHECKPOINT="/home/chuanruo/vggt_train/vggt_checkpoints/model.pt"
FRAMES="1 2 7 9 12 14 17 23 28"
TAG="vggt"
IMG_SIZE=518

DIRS=(
  "/home/chuanruo/TGGT/out/coat_run_20260705_051742/"                                       
  "/home/chuanruo/TGGT/out/coat_hanger_run_20260705_051742/"                               
  "/home/chuanruo/TGGT/out/coatrack_run_20260705_051742/"                                   
  "/home/chuanruo/TGGT/out/cockroach_run_20260705_051742/"                                  
  "/home/chuanruo/TGGT/out/cocoa_(beverage)_run_20260705_051742/"                           
  "/home/chuanruo/TGGT/out/coffee_maker_run_20260705_051742/"                               
  "/home/chuanruo/TGGT/out/coffee_table_run_20260705_051742/"                               
  "/home/chuanruo/TGGT/out/coffeepot_run_20260705_051742/"                                  
  "/home/chuanruo/TGGT/out/coil_run_20260705_051742/"
)

for DIR in "${DIRS[@]}"; do
  echo "Processing: $DIR"
  python export_to_viewer.py \
    --image_folder "$DIR/subsets/my_data_40" \
    --checkpoint "$CHECKPOINT" \
    --tags "$TAG" \
    --frames $FRAMES \
    --im $IMG_SIZE \
    --output results
  echo "---"
done

echo "Done!"
