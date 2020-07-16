#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J figure_12_EMCEE_b
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/EMCEE/figure_12_EMCEE_b_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/EMCEE/figure_12_EMCEE_b_warnings.stderr

python3 /home/astro/magnan/Repository_Stage_3A/scripts_EMCEE/draw_figure_12_EMCEE_b.py
