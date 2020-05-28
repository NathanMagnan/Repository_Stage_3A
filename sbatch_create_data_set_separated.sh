#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J create_data_set_separated
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/create_data_set_abacus_separated_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/create_data_set_abacus_separated_warnings.stderr

python3 /home/astro/magnan/Repository_Stage_3A/create_data_set_abacus_separated.py
