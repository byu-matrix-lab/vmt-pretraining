#!/bin/bash

#SBATCH --time=08:00:00   # walltime (8 hours)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=60G   # RAM per CPU core
#SBATCH -J "VATEX-eval"   # job name
#SBATCH --gpus=1
# # SBATCH --qos=cs
# # SBATCH --partition=cs


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
# export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source ~/fsl_groups/grp_mtlab/compute/environments/onmt/bin/activate

output_path=../../compute/data/outputs/none-tran/vatex-finetune/

sacrebleu ../../compute/vatex_baseline/Video-guided-Machine-Translation/results/masked1/tgt.txt -i ../../compute/vatex_baseline/Video-guided-Machine-Translation/results/masked1/preds.txt -m bleu chrf ter > ../../compute/vatex_baseline/Video-guided-Machine-Translation/results/masked1/metrics.txt --tokenize zh
# comet-score -s ../../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/src.txt -t ../../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/preds.txt -r ../../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/tgt.txt --gpus 0 --quiet > ../../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/comet_metrics.txt
