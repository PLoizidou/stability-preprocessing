#!/bin/bash

#SBATCH --time=06:00:00                     # run job for max 6 hours
#SBATCH --nodes=1                           # number of nodes (1 node = 1 computer)
#SBATCH --ntasks=1                         # number of independent jobs to run in parallel (i.e. tasks)
#SBATCH --cpus-per-task=32                   # number of cores per task
#SBATCH --mem-per-cpu=6G                    # memory per CPU core

#SBATCH -J "min1pipe"                         # job name
#SBATCH --mail-user=abrotman@caltech.edu    # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

##SBATCH -o /central/home/$USER/slurmout/slurm.%N.%j.out # STDOUT
##SBATCH -e /central/home/$USER/slurmout/slurm.%N.%j.err # STDERR

source activate stability-preprocessing
module load matlab/r2024a

DATA_FILE_STEM="miniscope2023-06-22T00_22_24_truncated"
DATA_FILE="${DATA_FILE_STEM}.avi"
DATA_DIR="/home/abrotman/stability-preprocessing/data"
TMP_DATA_DIR="/central/scratch/$USER/min1pipe/data"
TMP_DATA_FILE="${DATA_FILE_STEM}.tif"

MIN1PIPE="/home/abrotman/stability-preprocessing/MIN1PIPE"

mkdir -p $TMP_DATA_DIR

if [ ! -f "${DATA_DIR}/${DATA_FILE}" ]; then
    echo "Data file not found: ${DATA_DIR}/${DATA_FILE}"
    exit 1
fi

python scripts/avi_to_tiff.py --input_file "${DATA_DIR}/${DATA_FILE}" --output_file "${TMP_DATA_DIR}/${TMP_DATA_FILE}"

echo "Running MIN1PIPE preproc on ${TMP_DATA_DIR}/${TMP_DATA_FILE}"
matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('/home/abrotman/stability-preprocessing/scripts/min1pipe/')); preproc_min1pipe('${TMP_DATA_FILE}','${TMP_DATA_DIR}/','$MIN1PIPE'); exit;"