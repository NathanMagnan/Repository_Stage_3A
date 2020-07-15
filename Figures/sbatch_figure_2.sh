#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J figure_2
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/figure_2_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/figure_2_warnings.stderr

python3 /home/astro/magnan/Repository_Stage_3A/draw_figure_2.py
