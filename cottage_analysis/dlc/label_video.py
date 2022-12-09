import os

# os.environ['DLClight'] = 'True'
import sys
import matplotlib

matplotlib.use("Agg")  # make sure we use a backend that can run in headless mode


if __name__ == "__main__":
    import socket
    import deeplabcut

    hostname = socket.gethostname()
    print("Running on %s" % hostname)
    if hostname == "C02Z85AULVDC":
        # that's my laptop
        ROOT_DIR = "/Volumes/lab-znamenskiyp/home/shared/projects"
    else:
        # should be on camp
        ROOT_DIR = "/camp/lab/znamenskiyp/home/shared/projects"

    model_folder = "DLC_models/wehrcam_eye_tracking_2022"
    project = "blota_onix_pilote"
    mouse = "BRAC6692.4a"
    session = "S20221125"
    recordings = ["R154923", "R153834"]
    camera = "eye_camera"
    config_file = os.path.join(ROOT_DIR, model_folder, "config.yaml")
    from pathlib import Path

    sess = Path(ROOT_DIR) / project / mouse / session
    for rec in recordings:
        cam_dir = sess / rec / camera
        assert cam_dir.is_dir()
        save_path = cam_dir / "dlc_output"
        save_path.mkdir(exist_ok=True)
        for video in cam_dir.glob("{0}*_cropped.mp4".format(camera)):
            print("Doing %s" % video)

            print("Evaluate")
            deeplabcut.analyze_videos(
                config_file, [str(video)], save_as_csv=True, destfolder=save_path
            )
            print("Filtering")
            destfolder = cam_dir / ("labeled_{0}".format(filt))
            destfolder.mkdir(exist_ok=True)
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
                destfolder=save_path,
            )
            # print("Plot trajectory")
            # deeplabcut.plot_trajectories(config_file, [str(video)], track_method=None)
            print("Label videos")
            for use_filter in [False, True]:
                deeplabcut.create_labeled_video(
                    config_file,
                    [str(video)],
                    videotype=".mp4",
                    shuffle=1,
                    trainingsetindex=0,
                    filtered=use_filter,
                    fastmode=True,
                    save_frames=False,
                    keypoints_only=False,
                    Frames2plot=None,
                    displayedbodyparts="all",
                    displayedindividuals="all",
                    codec="mp4v",
                    outputframerate=None,
                    destfolder=save_path,
                    draw_skeleton=True,
                    trailpoints=0,
                    displaycropped=False,
                    color_by="bodypart",
                    modelprefix="",
                    track_method="",
                )
            print("Next")
    print("Done")
