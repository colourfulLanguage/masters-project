#!/bin/bash
#SBATCH --account=FUAL8_NWPAV
#SBATCH --nodes=1
#SBATCH --partition=boost_fua_prod
#SBATCH -t 24:00:00
#SBATCH --gres gpu:4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=jamie_batch_job_test

module load python/3.10.8--gcc--8.5.0
module unload anaconda3/2023.09-0
module load anaconda3/2023.09-0

source $HOME/.bashrc

VENV_NAME=ferminet_af
source activate $HOME/venvs/${VENV_NAME}/

python methyl_train_andres.py