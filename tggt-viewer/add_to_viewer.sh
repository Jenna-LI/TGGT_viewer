#!/bin/bash
#
# Add a new object to the TGGT Viewer
#
# Usage:
#   ./add_to_viewer.sh <checkpoint> <data_path> <tags...>
#
# Examples:
#   ./add_to_viewer.sh /path/to/model.pt /path/to/coat vggt
#   ./add_to_viewer.sh /path/to/model.pt /path/to/coat unseen vggt
#
# Arguments:
#   checkpoint  - Path to the model checkpoint (.pt file)
#   data_path   - Path to the data folder (with images/ subfolder)
#   tags        - One or more tags for filtering (e.g., "vggt", "unseen vggt")
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ "$#" -lt 3 ]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo ""
    echo "Usage: ./add_to_viewer.sh <checkpoint> <data_path> <tags...>"
    echo ""
    echo "Arguments:"
    echo "  checkpoint  - Path to the model checkpoint (.pt file)"
    echo "  data_path   - Path to the data folder"
    echo "  tags        - One or more tags (e.g., 'vggt' or 'unseen vggt')"
    echo ""
    echo "Examples:"
    echo "  ./add_to_viewer.sh /path/to/model.pt /path/to/coat vggt"
    echo "  ./add_to_viewer.sh /path/to/model.pt /path/to/coat unseen vggt"
    exit 1
fi

CHECKPOINT="$1"
DATA_PATH="$2"
shift 2
TAGS="$@"  # All remaining arguments are tags

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi

# Validate data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}Error: Data path not found: $DATA_PATH${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/results"

# Activate conda environment
echo -e "${GREEN}Activating vggt environment...${NC}"
source /home/chuanruo/anaconda3/etc/profile.d/conda.sh
conda activate vggt

# Show what we're doing
echo ""
echo -e "${GREEN}=== Adding to TGGT Viewer ===${NC}"
echo "Checkpoint: $CHECKPOINT"
echo "Data path:  $DATA_PATH"
echo "Tags:       $TAGS"
echo ""

# Run export
echo -e "${GREEN}Running export...${NC}"
cd "$SCRIPT_DIR"
python export_to_viewer.py \
    --data "$DATA_PATH" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_DIR" \
    --epoch 0 \
    --device cpu \
    --tags $TAGS

echo ""
echo -e "${GREEN}Done!${NC}"
echo "View results at: http://localhost:8080"
