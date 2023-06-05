#!/bin/bash

#SBATCH --time=04:00:00   # walltime (3 days, the maximum)
#SBATCH --ntasks-per-node=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=50G   # RAM per CPU core
#SBATCH -J "eval-none"   # job name
#SBATCH --gpus=1

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniconda3
source ~/.bashrc
conda activate 601final

python3 run_inference.py

output_path=../../compute/data/outputs/none-tran/vatex-finetune/

echo Running sacrebleu
sacrebleu ${output_path}tgts.txt -i ${output_path}preds.txt \
-m bleu chrf ter --tokenize zh > ${output_path}metrics.txt

echo Running comet
comet-score -s ${output_path}srcs.txt -t ${output_path}preds.txt \
-r ${output_path}tgts.txt --quiet \
--model ../../compute/comet/wmt22-comet-da/checkpoints/model.ckpt \
> ${output_path}comet_metrics.txt 


