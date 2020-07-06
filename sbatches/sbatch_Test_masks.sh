#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J test_masks
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/test_masks_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/test_masks_warnings.stderr

python3 /home/astro/magnan/Repository_Stage_3A/Test_masks.py
