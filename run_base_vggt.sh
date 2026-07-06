#!/bin/bash
# Run export_to_viewer.py with base VGGT checkpoint (518x518) for the specified directories

CHECKPOINT="/home/chuanruo/vggt_train/vggt_checkpoints/model.pt"
OUTPUT_DIR="/home/chuanruo/TGGT_viewer/tggt-viewer/results"
SCRIPT="/home/chuanruo/TGGT_viewer/tggt-viewer/export_to_viewer.py"

# Directories to process
RUNS=(
    "/home/chuanruo/TGGT/out/coat_run_20260703_041054"
    "/home/chuanruo/TGGT/out/coat_hanger_run_20260703_041054"
    "/home/chuanruo/TGGT/out/coatrack_run_20260703_041054"
    "/home/chuanruo/TGGT/out/cockroach_run_20260703_041054"
    "/home/chuanruo/TGGT/out/cocoa_(beverage)_run_20260703_041054"
    "/home/chuanruo/TGGT/out/coffee_maker_run_20260703_041054"
    "/home/chuanruo/TGGT/out/coffee_table_run_20260703_041054"
    "/home/chuanruo/TGGT/out/coffeepot_run_20260703_042934"
    "/home/chuanruo/TGGT/out/coil_run_20260703_042936"
)

cd /home/chuanruo/TGGT_viewer/tggt-viewer

for run_dir in "${RUNS[@]}"; do
    echo "=============================================="
    echo "Processing: $run_dir"
    echo "=============================================="

    # Find all my_data_* directories (excluding _annotations)
    for data_dir in "$run_dir/subsets"/my_data_*; do
        # Skip annotation directories
        if [[ "$data_dir" == *"_annotations" ]]; then
            continue
        fi

        # Skip if not a directory
        if [[ ! -d "$data_dir" ]]; then
            continue
        fi

        echo "  -> $data_dir"

        CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
            --data "$data_dir" \
            --checkpoint "$CHECKPOINT" \
            --output "$OUTPUT_DIR" \
            --device cuda \
            --tags vggt_base
    done
done

echo ""
echo "Done! Results saved to: $OUTPUT_DIR"
