#!/bin/bash
#SBATCH --job-name=assignment_1
#SBATCH --output=TESTassignment_1.out
#SBATCH --error=TESTassignment_1.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --partition=SCSEGPU_UG

module load anaconda
conda activate tf
python 1a_1.py

