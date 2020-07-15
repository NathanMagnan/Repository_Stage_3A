#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J create_data_set_Abacus_4
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/create_data_set_abacus_2_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/create_data_set_abacus_2_warnings.stderr

python3 /home/astro/magnan/Repository_Stage_3A/create_data_set_abacus_2.py
