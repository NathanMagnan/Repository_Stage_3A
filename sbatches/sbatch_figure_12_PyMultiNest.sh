#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J figure_12_PyMultiNest
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/figure_12_PyMultiNest_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/figure_12_PyMultiNest_warnings.stderr

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/epfl/atamone/Software/MultiNest/lib

python3 /home/astro/magnan/Repository_Stage_3A/data_figure_12_PyMultiNest.py
