from pathlib import Path
import deeplabcut

model = "headfixed_track_eye"
video = "/nem/"

config_path = Path("/nemo/lab/znamenskiyp/home/shared/projects/DLC_models/")
config_path /= f"{model}/config.yaml"


deeplabcut.create_labeled_video(config_path, video, videotype=".mp4")