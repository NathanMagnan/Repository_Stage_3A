#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J find_best_kernel
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/find_best_kernel_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/find_best_kernel_warnings.stderr

echo "OK, it worked"

python3 /home/astro/magnan/Repository_Stage_3A/find_best_kernel.py
