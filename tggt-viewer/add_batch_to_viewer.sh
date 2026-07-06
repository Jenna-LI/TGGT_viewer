#!/bin/bash
#
# Add multiple objects to the TGGT Viewer
#
# Usage:
#   ./add_batch_to_viewer.sh <checkpoint> <tag> <data_path1> [data_path2] ...
#
# Examples:
#   ./add_batch_to_viewer.sh /path/to/model.pt vggt /path/to/coat /path/to/hat /path/to/shoe
#   ./add_batch_to_viewer.sh /path/to/checkpoint.pt propegatt /path/to/run/subsets/*
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ "$#" -lt 3 ]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo ""
    echo "Usage: ./add_batch_to_viewer.sh <checkpoint> <tag> <data_path1> [data_path2] ..."
    echo ""
    echo "Examples:"
    echo "  ./add_batch_to_viewer.sh /path/to/model.pt vggt /path/to/coat /path/to/hat"
    echo "  ./add_batch_to_viewer.sh /path/to/checkpoint.pt propegatt /path/to/run/subsets/*"
    exit 1
fi

CHECKPOINT="$1"
TAG="$2"
shift 2
DATA_PATHS=("$@")

# Validate checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/results"

# Activate conda
echo -e "${GREEN}Activating vggt environment...${NC}"
source /home/chuanruo/anaconda3/etc/profile.d/conda.sh
conda activate vggt

cd "$SCRIPT_DIR"

TOTAL=${#DATA_PATHS[@]}
COUNT=0
SUCCESS=0
FAILED=0

echo ""
echo -e "${GREEN}=== Batch Adding to TGGT Viewer ===${NC}"
echo "Checkpoint: $CHECKPOINT"
echo "Tag:        $TAG"
echo "Total:      $TOTAL objects"
echo ""

for DATA_PATH in "${DATA_PATHS[@]}"; do
    COUNT=$((COUNT + 1))

    # Skip if not a directory
    if [ ! -d "$DATA_PATH" ]; then
        echo -e "[$COUNT/$TOTAL] ${YELLOW}SKIP${NC} (not a directory): $DATA_PATH"
        continue
    fi

    # Extract object name from path
    OBJ_NAME=$(basename "$DATA_PATH")

    echo -e "[$COUNT/$TOTAL] ${GREEN}Processing:${NC} $OBJ_NAME"

    if python export_to_viewer.py \
        --data "$DATA_PATH" \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT_DIR" \
        --epoch 0 \
        --device cpu \
        --object-id "$OBJ_NAME" \
        --tags "$TAG" 2>&1 | grep -E "^(Loading|Processing|Object:|Loaded|Running|Exporting|Done)"; then
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "  ${RED}Failed${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo -e "${GREEN}=== Complete ===${NC}"
echo "Success: $SUCCESS"
echo "Failed:  $FAILED"
echo ""
echo "View results at: http://localhost:8080"
echo "Filter by '$TAG' to see your new data."
