from pathlib import Path
import yaml
import shutil
import pandas as pd


def fix_bad_path_handling(config_file, user):
    """Fix dirty file path generated by DLC-GUI

    Mixing OS induces bugs with filepaths that are not easily parsed. Try to fix bad
    csv and h5 generated by manual labels like that.

    Issue is there:
    https://github.com/DeepLabCut/napari-deeplabcut/issues/30#issue-1301728053

    Fix is described here
    https://github.com/DeepLabCut/DeepLabCut/issues/2072

    Args:
        config_file (str): path to the config.yml file
        user (str): Name of the user who did the labeling. 
    """
    with open(config_file, "r") as fhandle:
        config = yaml.safe_load(fhandle)
    root = Path(config["project_path"])
    assert root.is_dir()

    videos = [Path(k).stem for k in config["video_sets"]]

    labeled = root / "labeled-data"
    for subfolder in labeled.glob("*"):
        if not subfolder.is_dir():
            continue
        if subfolder.name in videos:
            labels = subfolder / f"CollectedData_{user}.h5"
            if not labels.exists():
                continue
            df = pd.read_hdf(labels)
            if df.index.nlevels != 1:
                continue
            print(f"{subfolder.name} needs fixing")
            shutil.copy(labels, labels.with_suffix(".h5.backup"))
            shutil.copy(labels.with_suffix(".csv"), labels.with_suffix(".csv.backup"))
            fixed = pd.concat({subfolder.name: df}, names=[])
            fixed = pd.concat({"labeled-data": fixed}, names=[])
            fixed.to_hdf(labels.with_suffix(".h5"), key="data", mode="w")
            fixed.to_csv(labels.with_suffix(".csv"))


if __name__ == "__main__":
    config_file = "/nemo/lab/znamenskiyp/home/shared/projects/DLC_models/all_eyes_2023/config.yaml"
    fix_bad_path_handling(config_file)