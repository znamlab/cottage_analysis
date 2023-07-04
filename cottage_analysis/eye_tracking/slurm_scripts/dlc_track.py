"""This script is intended to be copied and edited by slurm_job.slurm_dlc_pupil"""
from cottage_analysis.eye_tracking.eye_tracking import dlc_pupil

camera_ds_id = "XXX_CAMERA_DS_ID_XXX"
model_name = "XXX_MODEL_NAME_XXX"
origin_id = "XXX_ORIGIN_ID_XXX"
project = "XXX_PROJECT_XXX"
conflicts = "XXX_CONFLICTS_XXX"
crop = "XXX_CROP_XXX"

if not isinstance(crop, bool):
    crop = False

dlc_pupil(
    camera_ds_id=camera_ds_id,
    model_name=model_name,
    origin_id=origin_id,
    project=project,
    crop=crop,
    conflicts=conflicts,
)
