#!/bin/bash

#SBATCH --time=04:00:00                     # run job for max 4 hours
#SBATCH --nodes=1                           # number of nodes (1 node = 1 computer)
#SBATCH --ntasks=16                         # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=6G                    # memory per CPU core

#SBATCH -J "caiman"                         # job name
#SBATCH --mail-user=abrotman@caltech.edu    # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

##SBATCH -o /central/home/$USER/slurmout/slurm.%N.%j.out # STDOUT
##SBATCH -e /central/home/$USER/slurmout/slurm.%N.%j.err # STDERR

# If you require a simple way to access an entire nodes resources
# including memory, use the exclusive switch. This could possibly
# extend your queue wait time as an entire node will need to be
# free before job executes. 

##SBATCH --exclusive


module load matlab/r2024a
matlab -nodisplay -nosplash -nodesktop -r "run('scripts/min1pipe/preproc_min1pipe.m'); exit;"