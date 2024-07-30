#!/bin/bash
#
#SBATCH --job-name=2p_exp_preprocess
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/logs/2p_analysis_%j.log"

ml purge

ml Anaconda3/2020.07
source activate base

conda activate 2p_analysis_cottage

echo start job
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/cottage_analysis/preprocessing/"
python quick_check.py