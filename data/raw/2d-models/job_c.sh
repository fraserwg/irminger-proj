#!/bin/bash
# Slurm job options (job-name, compute nodes, job time)
#SBATCH --array=6,8,16,19,21,28,31,32,34,36,37,38,41,44,46,47,48
#SBATCH --job-name=CIrm
#SBATCH --time=24:00:00
#SBATCH --nodes=16
#SBATCH --tasks-per-node=75
#SBATCH --cpus-per-task=1
#SBATCH --qos=taskfarm
#SBATCH --account=n01-SiAMOC
#SBATCH --partition=standard

# Fulls list 6-9,11-14,16-19,21-24,26-29,31-34,36-39,41-44,46-49

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
export RUN_DIR=run"$SLURM_ARRAY_TASK_ID"_c
export NEXT_RUN_DIR=run"$SLURM_ARRAY_TASK_ID"_d

cd ./$RUN_DIR
srun --distribution=block:block --hint=nomultithread ./mitgcmuv

# Make next run directory
mkdir ../$NEXT_RUN_DIR
mkdir ../$NEXT_RUN_DIR/input

cd ../$NEXT_RUN_DIR/input

# Copy binary input files over to new directory
ln -s ../../"$RUN_DIR"/input/delta* ./
ln -s ../../"$RUN_DIR"/input/bathymetry.* ./
ln -s ../../"$RUN_DIR"/input/merid_wind_stress* ./
ln -s ../../"$RUN_DIR"/input/*_ref.* ./
ln -s ../../"$RUN_DIR"/input/*_bound.* ./

# Copy data input files over to new directory
cd ..
ln -s ../input_data_files_d/* ./

# Copy pickup files over
cp ../"$RUN_DIR"/pickup* ./
