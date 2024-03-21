#!/bin/bash

#SBATCH --time=1-00:00:00   # walltime (3 days, the maximum)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=128G   # RAM per CPU core
#SBATCH -J "MT $INSERT_HARGS_HERE"   # job name
#SBATCH --gpus=1
#SBATCH --qos=cs

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniconda3
source ~/.bashrc
conda activate vmt

cd ../..

python3 run_train.py $INSERT_ARGS_HERE
