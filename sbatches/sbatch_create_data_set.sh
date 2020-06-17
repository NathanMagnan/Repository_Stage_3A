#!/bin/bash

#SBATCH -p p3
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH -J create_data_set
#SBATCH --exclusive
#SBATCH -o /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/create_data_set_abacus_output.stdout
#SBATCH -e /home/astro/magnan/Repository_Stage_3A/outputs_and_warnings/create_data_set_abacus_warnings.stderr

echo "Connection made"

module load mpi4py
module unload anaconda
module load python/3.7.1

export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}

srun --mpi=pmi2 /home/astro/magnan/Repository_Stage_3A/create_data_set_abacus.py
