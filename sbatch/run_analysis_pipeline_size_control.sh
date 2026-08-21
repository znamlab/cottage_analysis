#!/bin/bash
#
#SBATCH --job-name=size
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate v1_depth_map
if [ -z ${ANATOMICAL_ONLY+x} ]; then
    ANATOMICAL_ONLY="True"
fi

if [ -z ${AST_NEUROPIL+x} ]; then
    AST_NEUROPIL="False"
fi
if [ -z ${USE_ANNOTATED+x} ]; then
    USE_ANNOTATED="False"
fi
if [ -z ${PROTOCOL_BASE+x} ]; then
    PROTOCOL_BASE="SizeControl"
fi

echo Use anatomical only datasets: ${ANATOMICAL_ONLY}
echo Use ASt neuropil correction: ${AST_NEUROPIL}
echo Use annotated dataset: ${USE_ANNOTATED}
echo Protocol base: ${PROTOCOL_BASE}
cd "/camp/lab/znamenskiyp/home/users/blota/code/cottage_analysis/cottage_analysis/pipelines/"
# NB: defopt 6.4.0 exposes every parameter of main() as a POSITIONAL argument, bools
# included -- there are no --flags (`--help` lists only -h). The previous
# `--no-anatomical-only` / `--ast-neuropil` / `--use-annotated` echoes would have been
# rejected as unrecognised arguments; they never fired only because the current config
# leaves all three at their defaults, so each `$(...)` expanded to an empty string.
# Pass them positionally instead, in signature order.
python analysis_pipeline_size_control.py ${PROJECT} ${SESSION_NAME} ${CONFLICTS} \
    ${PHOTODIODE_PROTOCOL} ${USE_SLURM} ${ANATOMICAL_ONLY} ${AST_NEUROPIL} \
    ${USE_ANNOTATED} ${PROTOCOL_BASE}
