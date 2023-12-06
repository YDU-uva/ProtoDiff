#!/bin/sh

#SBATCH -n 18
#SBATCH -p gpu
#SBATCH --time=36:00:00

#module load miniconda3
##
#conda init bash
#
#source activate
#
#conda activate py38

python train_1.py --gpu 0
python train_5.py --gpu 0