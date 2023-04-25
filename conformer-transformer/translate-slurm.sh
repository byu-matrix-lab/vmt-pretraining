#!/bin/bash

#SBATCH --time=08:00:00   # walltime (8 hours)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=60G   # RAM per CPU core
#SBATCH -J "VATEX-eval1"   # job name
#SBATCH --gpus=1
#SBATCH --qos=cs
#SBATCH --partition=cs


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniconda3
source ~/.bashrc
conda activate 601final

# nvidia-smi

python3 translate.py
> ../../compute/models/conformer-transformer-vatex/dmodel_run1/output/metrics.txt
sacrebleu ../../compute/models/conformer-transformer-vatex/dmodel_run1/output/tgt.txt -i ../../compute/models/conformer-transformer-vatex/dmodel_run1/output/preds.txt -m bleu chrf ter > ../../compute/models/conformer-transformer-vatex/dmodel_run1/output/metrics.txt --tokenize zh
# comet-score -s ../../compute/models/conformer-transformer-vatex/run6/output/src.txt -t ../../compute/models/conformer-transformer-vatex/run6/output/preds.txt -r ../../compute/models/conformer-transformer-vatex/run6/output/tgt.txt >> ../../compute/models/conformer-transformer-vatex/run6/output/metrics.txt
