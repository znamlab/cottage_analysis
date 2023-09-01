"""This script is intended to be copied and edited by slurm_job.fit_ellipses"""
from cottage_analysis.eye_tracking import eye_tracking

camera_ds_id = "XXX_CAMERA_DS_ID_XXX"
project_id = "XXX_PROJECT_ID_XXX"
likelihood = "XXX_LIKELIHOOD_XXX"

eye_tracking.fit_ellipse(
    camera_ds_id=camera_ds_id, project_id=project_id, likelihood_threshold=likelihood
)
