#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J figure_12_EMCEE_d
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/EMCEE/figure_12_EMCEE_d_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/EMCEE/figure_12_EMCEE_d_warnings.stderr

python3 /home/astro/magnan/Repository_Stage_3A/scripts_EMCEE/draw_figure_12_EMCEE_d.py
