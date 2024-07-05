#!/bin/bash

export CAIMAN_DATA=/media/toor/Elements/caiman_data

LINEAR_CONFIG_FILE="/home/toor/Desktop/linear-abrotman-2024-05-29/config.yaml"
HOME_CONFIG_FILE="/home/toor/Desktop/home-abrotman-2024-05-29/config.yaml"

LINEAR_REGEX="*resnet50_linear*_filtered.h5"
HOME_REGEX="*resnet50_home*_filtered.h5"

LINEAR_HDF5_KEY="behavior_linear"
HOME_HDF5_KEY="behavior_home"

LINEAR_POSE_KEY="pose_linear"
HOME_POSE_KEY="pose_home"

# Check if the session folder argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Session folder argument is missing."
    echo "Usage: $0 <session_folder>"
    exit 1
fi

# Assign the session folder argument to a variable
miniscope_file=$1
home_video_file=$2
linear_video_file=$3

# Find the path to Conda
CONDA_PATH=$(which conda)

# Check if Conda was found
if [ -z "$CONDA_PATH" ]; then
    echo "Error: Conda not found. Please ensure Conda is installed and available in your PATH."
    exit 1
fi

# Get the directory containing the Conda binary
CONDA_DIR=$(dirname "$CONDA_PATH")

# Initialize Conda
__conda_setup="$("$CONDA_PATH" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$CONDA_DIR/../etc/profile.d/conda.sh" ]; then
        . "$CONDA_DIR/../etc/profile.d/conda.sh"
    else
        export PATH="$CONDA_DIR:$PATH"
    fi
fi
unset __conda_setup

# Run DLC pose estimation
# TODO: Add similar configuration for the T-maze

conda activate dlc # Activate the DLC environment

python scripts/estimate_pose.py \
    $LINEAR_CONFIG_FILE \
    $linear_video_file \
    --gpu_id 0

python scripts/estimate_pose.py \
    $HOME_CONFIG_FILE \
    $home_video_file \
    --gpu_id 0

conda deactivate # Deactivate the DLC environment

conda activate stability-preprocessing # Activate the stability-preprocessing environment

# TODO: Uncomment this if saving Caiman to NWB gets figured out
# # Curate the pose data in NWB format
# python scripts/curate_pose_nwb.py \
#     $LINEAR_CONFIG_FILE \
#     $session_folder \
#     $LINEAR_HDF5_KEY \
#     $LINEAR_POSE_KEY \
#     --match_regex $LINEAR_REGEX

# python scripts/curate_pose_nwb.py \
#     $HOME_CONFIG_FILE \
#     $session_folder \
#     $HOME_HDF5_KEY \
#     $HOME_POSE_KEY \
#     --match_regex $HOME_REGEX

# Preprocess the session data
python scripts/preproc_caiman.py \
--input_path $miniscope_file  \
--min_corr 0.85 \
--min_pnr 6.5 \
--min_SNR 3 \
--rval_thr 0.8 \
--gnb 0

conda deactivate

