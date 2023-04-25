#!/bin/bash

#SBATCH --time=3-00:00:00   # walltime (3 days, the maximum)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=60G   # RAM per CPU core
#SBATCH -J "VATEX-contran"   # job name
#SBATCH --gpus=1
#SBATCH --qos=cs
#SBATCH --partition=cs


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source comet-env/bin/activate

nvidia-smi

comet-score -s ../../compute/models/conformer-transformer-vatex/run6/output/src.txt -t ../../compute/models/conformer-transformer-vatex/run6/output/preds.txt -r ../../compute/models/conformer-transformer-vatex/run6/output/tgt.txt >> ../../compute/models/conformer-transformer-vatex/run6/output/metrics.txt --model Unbabel/wmt22-comet-da --gpus 0
# comet-score -s ../../compute/models/text-only/run6/output/src.txt -t ../../compute/models/text-only/run6/output/preds.txt -r ../../compute/models/text-only/run6/output/tgt.txt >> ../../compute/models/text-only/run6/output/metrics.txt --model Unbabel/wmt22-comet-da
