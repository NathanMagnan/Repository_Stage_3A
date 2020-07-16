#!/bin/bash

#SBATCH -p s2
#SBATCH --ntasks=1
#SBATCH -J test_masks_temporary
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/test_masks_temporary_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/test_masks_temporary_warnings.stderr

python3 /home/astro/magnan/Repository_Stage_3A/Test_masks_temporary.py
