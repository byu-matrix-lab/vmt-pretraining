
cd jobs/train-mad
pwd
for file in *.sh; do
    echo ${file}
    sbatch ${file}
done
