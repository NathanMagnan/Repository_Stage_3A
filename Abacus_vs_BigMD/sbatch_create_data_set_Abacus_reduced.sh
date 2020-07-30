#!/bin/bash

#SBATCH -p p4
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=1
#SBATCH -J create_data_set_Abacus_reduced
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/Abacus_vs_BigMD/create_data_set_Abacus_reduced_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/Abacus_vs_BigMD/create_data_set_Abacus_reduced_warnings.stderr

module load mpi4py

srun -n 80 -c 1 --mpi=pmi2 python3 /home/astro/magnan/Repository_Stage_3A/Abacus_vs_BigMD/create_data_set_Abacus_reduced.py
