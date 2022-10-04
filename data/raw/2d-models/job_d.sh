#!/bin/bash
# Slurm job options (job-name, compute nodes, job time)
#SBATCH --array=6,8,16,19,21,28,31,32,34,36,37,38,41,44,46,47,48
#SBATCH --job-name=DIrm
#SBATCH --time=24:00:00
#SBATCH --nodes=16
#SBATCH --tasks-per-node=75
#SBATCH --cpus-per-task=1
#SBATCH --qos=taskfarm
#SBATCH --account=n01-SiAMOC
#SBATCH --partition=standard


# Setup the job environment (this module needs to be loaded before any other modules)
module purge
module load PrgEnv-cray

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1

# Launch the parallel job
#   Using 256 MPI processes and 128 MPI processes per node
#   srun picks up the distribution from the sbatch options
export RUN__DIR=run"$SLURM_ARRAY_TASK_ID"_d

cd ./$RUN__DIR
srun --distribution=block:block --hint=nomultithread ./mitgcmuv
