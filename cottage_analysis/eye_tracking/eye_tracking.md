# Eye tracking module

This module contains function used to track the eyes with DLC and fit ellipses.

## Module organization

The module is organized as follows:

- `eye_tracking.py`: contains the main functions and should be the only entry point for the user.
- `slurm_job.py`: contains the functions used to submit jobs to the cluster. 
- `slurm_scripts`: contains template scripts copied and modified by `slurm_job.py`.

## Tracking

The first step is to use `eye_tracking.run_dlc` on each camera dataset to track the eyes 
with DLC. This need to be run twice, once on the full uncropped video to find the eye
and once on the cropped video.

For now, this step is manual (too annoying to set up something that waits for the job to
finish and then submit the next one as the job needs to be finished to defined the parameters
of the next slurm job).

To run the tracking on slurm, you need conda environment called `dlc_nogui` with
deeplabcut and cottage_analysis installed. This was done separately from the main
conda environment as deeplabcut is a pain to install.

