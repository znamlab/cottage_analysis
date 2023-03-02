"""This script is intended to be copied and edited by find_pupil.dlc_track"""
import deeplabcut
from pathlib import Path

video = "XXX_VIDEO_XXX"
config_file = "XXX_MODEL_XXX"
target_folder = "XXX_TARGET_XXX"
filter = "XXX_FILTER_XXX"
label = "XXX_LABEL_XXX"
origin_id = "XXX_ORIGIN_ID_XXX"
project = "XXX_PROJECT_XXX"

if origin_id.startswith("XXX_"):
    origin_id = None

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
    destfolder=target_folder,
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

print("Analyzing", flush=True)
out = deeplabcut.analyze_videos(**analyse_kwargs)
target_folder = Path(target_folder)
video = Path(video)
dlc_output = target_folder / f"{video.stem}{out}.h5"
if not dlc_output.exists():
    raise IOError(f"DLC ran but I cannot find the output. {dlc_output} does not exist.")

# Adding to flexilims if origin_id is provided
if origin_id is not None:
    print("Updating flexilims", flush=True)
    # imports here to keep the rest independant
    import flexiznam as flz
    from flexiznam.schema import Dataset

    flm_sess = flz.get_flexilims_session(project)
    ds = Dataset.from_origin(
        origin_id=origin_id,
        dataset_type="dlc_tracking",
        flexilims_session=flm_sess,
        base_name=f"dlc_tracking_{video.stem}",
        conflicts="overwrite",
    )
    ds.extra_attributes = analyse_kwargs
    ds.path = dlc_output.relative_to(flz.PARAMETERS["data_root"]["processed"])
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
