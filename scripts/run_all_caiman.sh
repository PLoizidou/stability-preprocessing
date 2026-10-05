#!/bin/bash
# Run preproc_caiman_final.py on all TMaze_miniscope*.avi files under a given root directory
# Usage:
#   ./run_all_tmaze.sh /path/to/root_dir

# Take root directory from the first argument
ROOT_DIR="$1"

# If no argument given, show usage and exit
if [ -z "$ROOT_DIR" ]; then
    echo "Usage: $0 /path/to/root_dir"
    exit 1
fi

# Directory where the Python script is located
SCRIPT_DIR="/home/toor/Desktop/stability-preprocessing/scripts"

# Conda environment to use
ENV_NAME="stability-preprocessing"

echo "Using root directory: $ROOT_DIR"
echo "Script directory:     $SCRIPT_DIR"
echo "Conda environment:    $ENV_NAME"
echo

# Find all matching AVI files and process each one
find "$ROOT_DIR" -type f -iname "*Linear_miniscope*.avi" | while read -r avi_file; do
    echo "Processing: $avi_file"
    conda run -n "$ENV_NAME" python "$SCRIPT_DIR/preproc_caiman_final.py" --input_path "$avi_file"
done

