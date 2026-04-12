#!/bin/bash -l
#SBATCH --job-name=train_bce
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB|GPU_MEM:141GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu,akundaje,owners
#SBATCH --time=48:00:00

ml openblas/0.3.28
ml xsimd/8.1.0
ml xz/5.8.1
ml hdf5/1.14.4
ml arrow/22.0.0
ml load py-pyarrow/18.1.0_py312
ml lz4/1.8.0
ml biology
ml htslib
ml ucsc-utils

mamba activate torch
nvidia-smi -L
time python train_bce_SCREEN.py