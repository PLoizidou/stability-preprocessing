#!/bin/bash

#SBATCH --time=06:00:00                     # run job for max 4 hours
#SBATCH --nodes=1                           # number of nodes (1 node = 1 computer)
#SBATCH --ntasks=1                         # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=32                   # number of OpenMP threads
#SBATCH --mem-per-cpu=4G                    # memory per CPU core

#SBATCH -J "min1pipe"                         # job name
#SBATCH --mail-user=abrotman@caltech.edu    # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

##SBATCH -o /central/home/$USER/slurmout/slurm.%N.%j.out # STDOUT
##SBATCH -e /central/home/$USER/slurmout/slurm.%N.%j.err # STDERR


module load matlab/r2024a
matlab -nodisplay -nosplash -nodesktop -r "run('scripts/min1pipe/preproc_min1pipe.m'); exit;"