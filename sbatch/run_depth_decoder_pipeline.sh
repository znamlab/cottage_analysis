#!/bin/bash --login
#
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
conda activate v1_depth_map

echo Processing ${SESSION_NAME} in project ${PROJECT} with photodiode protocol ${PHOTODIODE_PROTOCOL} use slurm ${USE_SLURM}...
cd "/nemo/lab/znamenskiyp/home/users/znamenp/code/cottage_analysis/cottage_analysis/pipelines/"
python depth_decoder_pipeline.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} ${PHOTODIODE_PROTOCOL} ${USE_SLURM} ${ANATOMICAL_ONLY} ${AST_NEUROPIL}