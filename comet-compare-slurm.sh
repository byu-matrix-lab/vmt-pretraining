#!/bin/bash

#SBATCH --time=04:00:00   # walltime (3 days, the maximum)
#SBATCH --ntasks-per-node=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=50G   # RAM per CPU core
#SBATCH -J "comet-compare"   # job name
#SBATCH --gpus=1

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniconda3
source ~/.bashrc
conda activate vmt




# echo Running comet compare on pretraining with video vs not video
# comet-compare -s ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/src.txt \
# -t  ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/preds.txt ../compute/data/outputs/text_only_finetune/none_tran/vatex_finetune/preds.txt ../compute/data/outputs/none_tran/vatex_finetune/preds.txt ../compute/data/outputs/text_only_finetune/con_tran/vatex_finetune/preds.txt ../compute/data/outputs/con_tran/vatex_finetune/preds.txt ../compute/data/outputs/text_only_finetune/tran_tran/vatex_finetune/preds.txt ../compute/data/outputs/tran_tran/vatex_finetune/preds.txt \
# -r ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/tgt.txt \
# --model ../compute/comet/wmt22-comet-da/checkpoints/model.ckpt \
# > ../compute/data/outputs/comet_compare_no_mask.txt

echo Running comet compare on 60% masked
comet-compare -s ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/src.txt \
-t  ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/preds.txt  ../compute/data/outputs/text_only_finetune/tran_tran/mask_60_end_vatex_finetune/preds.txt ../compute/data/outputs/tran_tran/mask_60_end_vatex_finetune/preds.txt \
-r ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/tgt.txt \
--model ../compute/comet/wmt22-comet-da/checkpoints/model.ckpt \
> ../compute/data/outputs/comet_compare_60end_tran_tran.txt

# COMET no mask
echo Running comet on non-masked results
comet-compare -s ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/src.txt \
-t ../compute/data/outputs/con_tran/vatex_finetune/preds.txt ../compute/data/outputs/none_tran/vatex_finetune/preds.txt ../compute/data/outputs/tran_tran/vatex_finetune/preds.txt ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/preds.txt ../compute/data/outputs/con_tran/vatex_only/preds.txt ../compute/data/outputs/none_tran/vatex_only/preds.txt ../compute/data/outputs/tran_tran/vatex_only/preds.txt ../compute/data/outputs/con_tran/vatex_finetune_opus/preds.txt ../compute/data/outputs/none_tran/vatex_finetune_opus/preds.txt ../compute/data/outputs/tran_tran/vatex_finetune_opus/preds.txt \
-r ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/tgt.txt \
--model ../compute/comet/wmt22-comet-da/checkpoints/model.ckpt \
> ../compute/data/outputs/comet_compare_no_mask.txt

# # COMET 30% mask
# # TODO: Replace the vatex baseline with a masked version
# echo "Running comet with reference on 30% masked"
# comet-compare -s ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/src.txt \
# -t ../compute/data/outputs/con_tran/mask_30_rand_vatex_finetune/preds.txt ../compute/data/outputs/none_tran/mask_30_rand_vatex_finetune/preds.txt ../compute/data/outputs/tran_tran/mask_30_rand_vatex_finetune/preds.txt ../compute/vatex_baseline/Video-guided-Machine-Translation/results/mask_30_rand/preds.txt ../compute/data/outputs/con_tran/mask_30_rand_vatex_only/preds.txt ../compute/data/outputs/none_tran/mask_30_rand_vatex_only/preds.txt ../compute/data/outputs/tran_tran/mask_30_rand_vatex_only/preds.txt \
# -r ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/tgt.txt \
# --model ../compute/comet/wmt22-comet-da/checkpoints/model.ckpt \
# > ../compute/data/outputs/comet_compare_30_mask_rand.txt

# # COMET 60% mask
# # TODO: Replace the vatex baseline with a masked version
# echo "Running comet with reference on 60% masked"
# comet-compare -s ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/src.txt \
# -t ../compute/data/outputs/con_tran/mask_60_rand_vatex_finetune/preds.txt ../compute/data/outputs/none_tran/mask_60_rand_vatex_finetune/preds.txt ../compute/data/outputs/tran_tran/mask_60_rand_vatex_finetune/preds.txt ../compute/vatex_baseline/Video-guided-Machine-Translation/results/mask_60_rand/preds.txt ../compute/data/outputs/con_tran/mask_60_rand_vatex_only/preds.txt ../compute/data/outputs/none_tran/mask_60_rand_vatex_only/preds.txt ../compute/data/outputs/tran_tran/mask_60_rand_vatex_only/preds.txt \
# -r ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/tgt.txt \
# --model ../compute/comet/wmt22-comet-da/checkpoints/model.ckpt \
# > ../compute/data/outputs/comet_compare_60_mask_rand.txt

# # COMET Reference
# echo Running comet with reference
# comet-compare -s ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/src.txt \
# -t ../compute/data/outputs/con-tran/vatex-finetune/preds.txt ../compute/data/outputs/none-tran/vatex-finetune/preds.txt ../compute/data/outputs/tran-tran/vatex-finetune/preds.txt ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/preds.txt ../compute/data/outputs/con-tran/vatex-only/preds.txt ../compute/data/outputs/none-tran/vatex-only/preds.txt ../compute/data/outputs/tran-tran/vatex-only/preds.txt \
# -r ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/tgt.txt \
# --model ../compute/comet/wmt22-comet-da/checkpoints/model.ckpt \
# > ../compute/data/outputs/all_comet_metrics-new.txt

# # COMET Reference Free (COMET-QE)
# echo "Running comet without reference (COMET-QE)"
# comet-compare -s ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/src.txt \
# -t ../compute/data/outputs/con-tran/vatex-finetune/preds.txt ../compute/data/outputs/none-tran/vatex-finetune/preds.txt ../compute/data/outputs/tran-tran/vatex-finetune/preds.txt ../compute/vatex_baseline/Video-guided-Machine-Translation/results/testset/preds.txt \
# --model ../compute/comet/wmt22-comet-da/checkpoints/model.ckpt \
# > ../compute/data/outputs/all_comet_metrics.txt


