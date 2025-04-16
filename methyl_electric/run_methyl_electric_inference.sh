#!/bin/bash
#SBATCH -A FUAL8_NWPAV
#SBATCH -p boost_fua_prod
#SBATCH --time 24:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:4        # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=jamie_batch_job_test

srun python methyl_electric_inference.py