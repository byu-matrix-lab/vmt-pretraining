#!/bin/bash

#SBATCH --time=08:00:00   # walltime (3 days, the maximum)
#SBATCH --ntasks-per-node=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=50G   # RAM per CPU core
#SBATCH -J "comet-all"   # job name
#SBATCH --gpus=1

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniconda3
source ~/.bashrc
conda activate 601final

# output_path=../compute/data/outputs/con-tran/mask-60-end-vatex-only
# for output_path in ../compute/data/outputs/*/vatex-only-*; do
# for output_path in ../compute/data/outputs/*_*/*; do

for output_path in ../compute/vatex_baseline/Video-guided-Machine-Translation/results/*_*; do
    echo ${output_path}

    sacrebleu ${output_path}/tgt*.txt -i ${output_path}/preds.txt \
    -m bleu chrf ter --tokenize zh > ${output_path}/metrics.txt

    comet-score -s ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/src.txt \
    -r ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/tgt.txt \
    -t ${output_path}/preds.txt \
    --quiet \
    --model ../compute/comet/wmt22-comet-da/checkpoints/model.ckpt \
    > ${output_path}/comet_metrics.txt 
done



