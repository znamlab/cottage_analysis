from pathlib import Path
import deeplabcut

model = "headfixed_track_eye"
data_fold = Path("/camp/lab/znamenskiyp/home/shared/projects/blota_onix_pilote/")
data_fold /= "BRYA142.5d/S20231024/R140945_wehrcam_headmounted/"
video = data_fold / "eye_camera_deinterleaved_0/eye_camera_deinterleaved.mp4"

config_path = Path("/nemo/lab/znamenskiyp/home/shared/projects/DLC_models/")
config_path /= f"{model}/config.yaml"

deeplabcut.create_labeled_video(
    config_path,
    str(video),
    videotype=".mp4",
    destfolder=str(data_fold / "eye_camera_deinterleaved_0_dlc_tracking_cropped_0"),
)
