
cd jobs/finetune-vatex
pwd
for file in *.sh; do
    echo ${file}
    # sbatch ${file}
done
