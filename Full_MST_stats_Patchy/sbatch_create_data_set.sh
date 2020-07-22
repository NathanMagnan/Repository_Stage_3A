#!/bin/bash

#SBATCH -p p5
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH -J create_data_set_patchy
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Patchy/create_data_set_patchy.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Patchy/create_data_set_patchy.stderr

module load mpi4py

srun -n 32 -c 1 --mpi=pmi2 python3 /home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Patchy/Create_data_set_patchy.py
