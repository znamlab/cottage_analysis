#!/bin/bash
#SBATCH --job-name=dlc_eye_train
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL

echo 'ml-ing'
ml CUDA/10.0.130 cuDNN/7.5.0.56-CUDA-10.0.130 Anaconda3
echo 'Sourcing conda'
source /camp/apps/eb/software/Anaconda/conda.env.sh
echo 'activate'
conda activate /camp/lab/znamenskiyp/home/.conda/envs/DLC/

cd /camp/lab/znamenskiyp/home/users/blota/code/cottage_analysis
python cottage_analysis/dlc/train_network.py

