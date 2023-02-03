"""This script is intended to be copied and edited by find_pupil.dlc_track"""
import deeplabcut

video = "XXX_VIDEO_XXX"
config_file = "XXX_MODEL_XXX"
target_folder = "XXX_TARGET_XXX"
filter = "XXX_FILTER_XXX"
label = "XXX_LABEL_XXX"

print(f"Doing %s" % video)

print("Evaluate")
deeplabcut.analyze_videos(
    config_file, [str(video)], save_as_csv=True, destfolder=target_folder
)
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
