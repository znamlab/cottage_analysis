#!/bin/bash
#SBATCH --job-name=dlc_eye_train
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=blota@crick.ac.uk

echo 'ml-ing'
ml cuDNN/8.1.1.33-CUDA-11.2.1
echo 'Sourcing conda'
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
echo 'activate'
conda activate dlc_nogui

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/dlc_nogui/lib/

cd /camp/lab/znamenskiyp/home/users/blota/code/cottage_analysis
python cottage_analysis/dlc/train_network.py
