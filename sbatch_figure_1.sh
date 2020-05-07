#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J figure_1
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/figure_1_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/figure_1_warnings.stderr

python3 /home/astro/magnan/Repository_Stage_3A/figure_1.py
