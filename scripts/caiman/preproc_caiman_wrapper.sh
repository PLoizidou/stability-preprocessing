#!/bin/bash

#SBATCH --time=04:00:00                     # run job for max 4 hours
#SBATCH --nodes=1                           # number of nodes (1 node = 1 computer)
#SBATCH --ntasks=1                          # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=32                   # number of OpenMP threads
#SBATCH --mem-per-cpu=6G                    # memory per CPU core

#SBATCH -J "caiman"                         # job name
#SBATCH --mail-user=abrotman@caltech.edu    # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

##SBATCH -o ~/slurmout/slurm.caiman.%N.%j.out # STDOUT
##SBATCH -e ~/slurmout/slurm.caiman.%N.%j.err # STDERR

# If you require a simple way to access an entire nodes resources
# including memory, use the exclusive switch. This could possibly
# extend your queue wait time as an entire node will need to be
# free before job executes. 

##SBATCH --exclusive

INPUT_PATH=data/miniscope2023-06-22T00_22_24_truncated.avi
TMP_INPUT_PATH=/central/scratch/$USER/caiman
cp $INPUT_PATH $TMP_INPUT_PATH


source activate stability-preprocessing

python scripts/caiman/preproc_caiman.py \
    --input_path $TMP_INPUT_PATH/miniscope2023-06-22T00_22_24_truncated.avi \
    --output_path data/caiman_output \
    --log_severity WARNING \

    # dataset specific parameters
    --fr 25 \
    --decay_time 0.56 \
    --dxy 0.83 0.83 \

    # motion correction parameters
    --strides 64 64 \
    --overlaps 32 32 \
    --max_shifts 25 25 \
    --max_deviation_rigid 8 \
    --gSig_filt 8 8 \
    --pw_rigid True \

    # CNMF-E parameters
    --p 1 \
    --gSig 8 8 \
    --merge_thr 0.8 \
    --rf 32 \
    --stride_cnmf 16 \
    --ssub 1 \
    --tsub 3 \
    --gnb 0 \
    --min_corr 0.7 \
    --min_pnr 8 \
    --ssub_B 2 \

    # component evaluation parameters
    --min_SNR 1.5 \
    --rval_thr 0.75 \
    