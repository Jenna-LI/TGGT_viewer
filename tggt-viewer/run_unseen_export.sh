#!/bin/bash
# Export unseen objects to viewer format
# Usage: ./run_unseen_export.sh [device]
# device: cpu or cuda (default: cpu)

DEVICE=${1:-cpu}
CHECKPOINT="/home/chuanruo/vggt_train/training/logs/exp197/ckpts/checkpoint_190.pt"
EPOCH=190
OUTPUT_DIR="results"

# Unseen objects with their data paths (all have timestamp 134316)
declare -A UNSEEN_OBJECTS=(
    ["coat"]="/home/chuanruo/TGGT/out/coat_run_20260626_134316/subsets/my_data_32"
    ["coat_hanger"]="/home/chuanruo/TGGT/out/coat_hanger_run_20260626_134316/subsets/my_data_32"
    ["coatrack"]="/home/chuanruo/TGGT/out/coatrack_run_20260626_134316/subsets/my_data_32"
    ["cockroach"]="/home/chuanruo/TGGT/out/cockroach_run_20260626_134316/subsets/my_data_32"
    ["cocoa_(beverage)"]="/home/chuanruo/TGGT/out/cocoa_(beverage)_run_20260626_134316/subsets/my_data_32"
    ["coffee_maker"]="/home/chuanruo/TGGT/out/coffee_maker_run_20260626_134316/subsets/my_data_32"
    ["coffeepot"]="/home/chuanruo/TGGT/out/coffeepot_run_20260626_134316/subsets/my_data_32"
    ["coffee_table"]="/home/chuanruo/TGGT/out/coffee_table_run_20260626_134316/subsets/my_data_32"
    ["coil"]="/home/chuanruo/TGGT/out/coil_run_20260626_134316/subsets/my_data_32"
)

echo "=========================================="
echo "Exporting unseen objects to viewer"
echo "Device: $DEVICE"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

for obj in "${!UNSEEN_OBJECTS[@]}"; do
    data_path="${UNSEEN_OBJECTS[$obj]}"
    echo ""
    echo "Processing: $obj"
    echo "Data path: $data_path"

    if [ ! -d "$data_path" ]; then
        echo "  WARNING: Data path does not exist, skipping"
        continue
    fi

    python export_to_viewer.py \
        --data "$data_path" \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT_DIR" \
        --epoch $EPOCH \
        --device $DEVICE

    if [ $? -eq 0 ]; then
        echo "  SUCCESS: $obj exported"
    else
        echo "  ERROR: Failed to export $obj"
    fi
done

echo ""
echo "=========================================="
echo "Export complete!"
echo "=========================================="
