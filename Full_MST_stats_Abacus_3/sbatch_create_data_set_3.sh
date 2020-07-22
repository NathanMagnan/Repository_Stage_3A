#!/bin/bash

#SBATCH -p p4
#SBATCH --ntasks=21
#SBATCH --cpus-per-task=1
#SBATCH -J create_data_set_3
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus_3/create_data_set_3.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus_3/create_data_set_3.stderr

module load mpi4py

srun -n 21 -c 1 --mpi=pmi2 python3 /home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus_3/create_data_set_abacus_3.py
