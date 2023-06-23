
cd jobs/val
pwd
for file in *.sh; do
    echo ${file}
    sbatch ${file}
done
