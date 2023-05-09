"""This script is intended to be copied and edited by find_pupil.fit_ellipses"""

from pathlib import Path
from cottage_analysis.eye_tracking import eye_model_fitting

dlc_file = "XXX_DLC_FILE_XXX"
target_folder = "XXX_TARGET_XXX"
likelihood = "XXX_LIKELIHOOD_XXX"
label = "XXX_LABEL_XXX"
origin_id = "XXX_ORIGIN_ID_XXX"
project = "XXX_PROJECT_XXX"

if origin_id.startswith("XXX_"):
    origin_id = None
if likelihood.startswith("XXX_"):
    likelihood = None


dlc_file = Path(dlc_file)
assert dlc_file.exists()

target = Path(target_folder) / f"{dlc_file.stem}_ellipse_fits.csv"
print(f"Doing %s" % dlc_file)
ellipse_fits = eye_model_fitting.fit_ellipses(dlc_file, likelihood_threshold=likelihood)
print(f"Fitted, save to {target}")
ellipse_fits.to_csv(target, index=False)
print("Done")
