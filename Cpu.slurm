#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH -p compute
#SBATCH -A MANAPY-UM6P-ST-MSDA-1WABCJWE938-DEFAULT-CPU
#SBATCH --time=1-00:00:00 
#SBATCH --job-name test2
#SBATCH --output /home/houssam.bouchouk/lustre/manapy-um6p-st-msda-1wabcjwe938/users/houssam.bouchouk/cpu%j.log
# Load modules

source ~/.bashrc

module load Anaconda3

conda activate sklearn

python train.py --config config/adult.yml --model_name LinearModel