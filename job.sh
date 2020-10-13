#!/bin/bash
#SBATCH --job-name=assignment_1
#SBATCH --output=assignment_1.out
#SBATCH --error=assignment_1.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --partition=SCSEGPU_UG

module load anaconda
<<<<<<< HEAD
source activate tf2.2
=======
conda activate tf2.2
>>>>>>> 943410d1ec1ecbb85141f091cc97c9a6bd459854
python $1

