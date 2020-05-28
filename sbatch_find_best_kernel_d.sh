#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J find_best_kernel_d
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/find_best_kernel_d_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/find_best_kernel_d_warnings.stderr

echo "OK, it worked"

python3 /home/astro/magnan/Repository_Stage_3A/find_best_kernel_d.py
