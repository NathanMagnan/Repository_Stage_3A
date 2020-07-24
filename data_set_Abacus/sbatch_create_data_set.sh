#!/bin/bash

#SBATCH -p p4
#SBATCH --ntasks=61
#SBATCH --cpus-per-task=1
#SBATCH -J create_data_set
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/data_set_Abacus/create_data_set_abacus_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/data_set_Abacus/create_data_set_abacus_warnings.stderr

module load mpi4py

srun -n 61 -c 1 --mpi=pmi2 python3 /home/astro/magnan/Repository_Stage_3A/data_set_Abacus/create_data_set_abacus.py
