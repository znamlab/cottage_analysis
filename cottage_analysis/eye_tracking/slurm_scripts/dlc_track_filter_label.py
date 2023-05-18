"""This script is intended to be copied and edited by slurm_job.slurm_dlc_pupil"""
import deeplabcut
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

video = "XXX_VIDEO_XXX"
config_file = "XXX_MODEL_XXX"
target_folder = "XXX_TARGET_XXX"
filter = "XXX_FILTER_XXX"
label = "XXX_LABEL_XXX"
origin_id = "XXX_ORIGIN_ID_XXX"
project = "XXX_PROJECT_XXX"
crop_info = "XXX_CROP_INFO_XXX"

if isinstance(crop_info, str):
    crop_info = None
else:
    assert len(crop_info) == 4, "crop_info must be [xmin, xmax, ymin, ymax]"

if origin_id.startswith("XXX_"):
    origin_id = None
else:
    # set up flexilims but do not upload
    assert not project.startswith("XXX_"), "Project must be provided to use origin_id"
    assert isinstance(origin_id, str), "origin_id must be a string"
    assert len(origin_id) == 24, "origin_id must be 24 characters long"
    import flexiznam as flz
    from flexiznam.schema import Dataset

    flm_sess = flz.get_flexilims_session(project)

    ds = Dataset.from_origin(
        origin_id=origin_id,
        dataset_type="dlc_tracking",
        flexilims_session=flm_sess,
        conflicts="append",
    )
    # the conflict argument is not important here as we will change the dataset name.
    # conflict handling is done in slurm_job.py
    # dataset_name is changed by updating the last value of the genealogy list
    ds.genealogy = list(ds.genealogy[:-1]) + [Path(target_folder).name]
    ds.path = ds.path.with_name(Path(target_folder).name)  # update path to match change
    assert str(target_folder) == str(ds.path_full)

print(f"Doing %s" % video)

print("Evaluate")
analyse_kwargs = dict(
    config=config_file,
    videos=[str(video)],
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    gputouse=None,
    save_as_csv=False,
    in_random_order=True,
    destfolder=str(target_folder),
    batchsize=None,
    cropping=None,
    TFGPUinference=True,
    dynamic=(False, 0.5, 10),
    modelprefix="",
    robust_nframes=False,
    allow_growth=False,
    use_shelve=False,
    auto_track=True,
    n_tracks=None,
    calibrate=False,
    identity_only=False,
)

target_folder = Path(target_folder)
video = Path(video)

if crop_info is not None:
    assert len(crop_info) == 4, "crop_info must be [xmin, xmax, ymin, ymax]"
    analyse_kwargs["cropping"] = crop_info
    print(f'New crop: {analyse_kwargs["cropping"]}')

print("Analyzing", flush=True)
out = deeplabcut.analyze_videos(**analyse_kwargs)

dlc_output = target_folder / f"{video.stem}{out}.h5"
if not dlc_output.exists():
    raise IOError(f"DLC ran but I cannot find the output. {dlc_output} does not exist.")

# Adding to flexilims if origin_id is provided
if origin_id is not None:
    print("Updating flexilims", flush=True)
    ds.extra_attributes = dict(
        analyse_kwargs, dlc_prefix=out, dlc_file=f"{video.stem}{out}.h5"
    )
    ds.path = target_folder.relative_to(flz.PARAMETERS["data_root"]["processed"])
    ds.update_flexilims(mode="overwrite")


if filter:
    print("Filtering")
    deeplabcut.filterpredictions(
        str(config_file),
        [str(video)],
        shuffle=1,
        trainingsetindex=0,
        filtertype="arima",
        p_bound=0.01,
        ARdegree=3,
        MAdegree=1,
        alpha=0.01,
        save_as_csv=True,
        destfolder=target_folder,
    )

if label:
    print("Label videos")
    deeplabcut.create_labeled_video(
        config_file,
        [str(video)],
        videotype=".mp4",
        shuffle=1,
        trainingsetindex=0,
        filtered=filter,
        fastmode=True,
        save_frames=False,
        keypoints_only=False,
        Frames2plot=None,
        displayedbodyparts="all",
        displayedindividuals="all",
        codec="mp4v",
        outputframerate=None,
        destfolder=target_folder,
        draw_skeleton=True,
        trailpoints=0,
        displaycropped=False,
        color_by="bodypart",
        modelprefix="",
        track_method="",
    )
