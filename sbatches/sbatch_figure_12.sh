#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J figure_12
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/figure_12_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/figure_12_warnings.stderr

python3 /home/astro/magnan/Repository_Stage_3A/data_figure_12.py
