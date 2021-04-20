#!/bin/bash
#SBATCH --job-name=dlc_eye_train
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
ml CUDA/10.0.130 cuDNN/7.5.0.56-CUDA-10.0.130 Anaconda3
conda activate DLC
cd /camp/lab/znamenskiyp/home/user/blota/code/cottage_analysis
python -m cottage_analysis/dlc/train_network.py

