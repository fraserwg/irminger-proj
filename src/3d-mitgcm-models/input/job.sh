#!/bin/bash
# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=A3DIrmingerControl
#SBATCH --time=36:00:00
#SBATCH --nodes=10
#SBATCH --tasks-per-node=75
#SBATCH --cpus-per-task=1

# #SBATCH --qos=short
# #SBATCH --reservation=shortqos

#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --account=n01-SiAMOC

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
srun --distribution=block:block --hint=nomultithread ./mitgcmuv
