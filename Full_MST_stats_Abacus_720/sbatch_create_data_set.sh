#!/bin/bash

#SBATCH -p p4
#SBATCH --ntasks=41
#SBATCH --cpus-per-task=1
#SBATCH -J create_data_set_720
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus_720/create_data_set_720.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus_720/create_data_set_720.stderr

module load mpi4py

srun -n 41 -c 1 --mpi=pmi2 python3 /home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus_720/Create_data_set_abacus_720.py
