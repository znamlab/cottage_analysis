#!/bin/bash
#
#SBATCH --job-name=decoder
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --partition=ncpu
#SBATCH --mail-user=cypranc@crick.ac.uk
#SBATCH --mail-type=END,FAIL

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate v1_depth_seq

echo Processing ${SESSION_NAME} in project ${PROJECT} with photodiode protocol ${PHOTODIODE_PROTOCOL} use slurm ${USE_SLURM}...
cd "/nemo/lab/znamenskiyp/home/users/cypranc/cottage_analysis/cottage_analysis/pipelines/"
python depth_decoder_pipeline_roi_subset.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} ${PHOTODIODE_PROTOCOL} ${USE_SLURM} 
