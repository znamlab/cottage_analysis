#!/bin/bash
#
#SBATCH --job-name=2p_analysis
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/logs/2p_analysis_%j.log"
. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate 2p_analysis_cottage

echo Processing ${SESSION_NAME} in project ${PROJECT}...
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/cottage_analysis/cottage_analysis/pipelines/"
python sta_pipeline.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} ${PHOTODIODE_PROTOCOL} 