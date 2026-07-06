#!/bin/bash

# Activate vggt conda environment
source /home/chuanruo/anaconda3/etc/profile.d/conda.sh
conda activate vggt

CHECKPOINT="/home/chuanruo/vggt_train/vggt_checkpoints/model.pt"
OUTPUT="/home/chuanruo/TGGT_viewer/tggt-viewer/results"
EPOCH=0

# Object names and their directories
declare -A OBJECTS=(
  ["coat"]="/home/chuanruo/TGGT/out/raw_captures/coat"
  ["coat_hanger"]="/home/chuanruo/TGGT/out/raw_captures/coat_hanger"
  ["coatrack"]="/home/chuanruo/TGGT/out/raw_captures/coatrack"
  ["cockroach"]="/home/chuanruo/TGGT/out/raw_captures/cockroach"
  ["cocoa_(beverage)"]="/home/chuanruo/TGGT/out/raw_captures/cocoa_(beverage)"
  ["coffee_maker"]="/home/chuanruo/TGGT/out/raw_captures/coffee_maker"
  ["coffeepot"]="/home/chuanruo/TGGT/out/raw_captures/coffeepot"
  ["coffee_table"]="/home/chuanruo/TGGT/out/raw_captures/coffee_table"
  ["coil"]="/home/chuanruo/TGGT/out/raw_captures/coil"
)

cd /home/chuanruo/TGGT_viewer/tggt-viewer

# Process count
TOTAL=${#OBJECTS[@]}
COUNT=0

for OBJ_NAME in "${!OBJECTS[@]}"; do
  COUNT=$((COUNT + 1))
  DIR="${OBJECTS[$OBJ_NAME]}"
  IMAGES_DIR="${DIR}/images"

  if [ -d "$IMAGES_DIR" ]; then
    echo "[$COUNT/$TOTAL] Processing: $OBJ_NAME"
    python export_to_viewer.py \
      --data "$DIR" \
      --checkpoint "$CHECKPOINT" \
      --epoch $EPOCH \
      --output "$OUTPUT" \
      --device cpu \
      --object-id "$OBJ_NAME" \
      --tags vggt
  else
    echo "[$COUNT/$TOTAL] SKIP (no images): $OBJ_NAME"
  fi
done

echo "Done! Processed $COUNT directories."
