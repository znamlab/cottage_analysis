#!/bin/bash
#SBATCH --job-name=dlc_eye_train
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=eye_train.out
#SBATCH --error=eye_train.err
# #SBATCH --reservation=cuda12
#SBATCH --mail-user=blota@crick.ac.uk

# echo 'ml-ing'
ml cuDNN/8.4.1.50-CUDA-11.7.0 
echo 'Sourcing conda'
ml Anaconda3
source activate base
echo 'activate'
conda activate dlc_nogui

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/dlc_nogui/lib/

cd /camp/lab/znamenskiyp/home/users/blota/code/cottage_analysis
python cottage_analysis/dlc/train_network.py

