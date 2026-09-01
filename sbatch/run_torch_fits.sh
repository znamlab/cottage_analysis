#!/bin/bash
#
#SBATCH --job-name=cottage_analysis
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
conda activate v1_depth_seq

echo Processing ${SESSION_NAME} in project ${PROJECT} with photodiode protocol ${PHOTODIODE_PROTOCOL} use slurm ${USE_SLURM}...
# set defaut values for optional arguments
if [ -z ${CONFLICTS+x} ]; then
    CONFLICTS="skip"
fi
if [ -z ${USE_SLURM+x} ]; then
    USE_SLURM="True"
fi
if [ -z ${PROTOCOL_BASE+x} ]; then
    PROTOCOL_BASE="SpheresPermTubeReward"
fi
if [ -z ${ANATOMICAL_ONLY+x} ]; then
    ANATOMICAL_ONLY="True"
fi
if [ -z ${AST_NEUROPIL+x} ]; then
    AST_NEUROPIL="False"
fi
if [ -z ${USE_ANNOTATED+x} ]; then
    USE_ANNOTATED="False"
fi

echo Use ${PROTOCOL_BASE}
echo Use anatomical only datasets: ${ANATOMICAL_ONLY}
echo Use ASt neuropil correction: ${AST_NEUROPIL}
echo Use annotated dataset: ${USE_ANNOTATED}
cd "/nemo/lab/znamenskiyp/home/users/cypranc/cottage_analysis/cottage_analysis/pipelines"
python analysis_pipeline_torch.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} ${PHOTODIODE_PROTOCOL} ${USE_SLURM} ${PROTOCOL_BASE} ${ANATOMICAL_ONLY} ${AST_NEUROPIL} ${USE_ANNOTATED}
