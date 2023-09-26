"""Function to preprocess minicam data."""

from functools import partial
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import flexiznam as flz
from flexiznam.schema import CameraData
from znamutils import slurm_it
from cottage_analysis.io_module import video
import yaml


def run_deinterleave(camera_ds, conflicts="abort", use_slurm=True, dependency=None):
    """Run deinterleave on a camera dataset.

    Args:
        camera_ds (flexiznam.Dataset): camera dataset
        conflicts (str, optional): How to handle conflicts. Can be "abort", "skip" or
            "overwrite". Defaults to "abort".
        use_slurm (bool, optional): whether to use slurm. Defaults to True.
        dependency (str, optional): dependency for slurm. Defaults to None.

    Returns:
        str: job id if use_slurm is True
        str: path to the deinterleaved video
    """
    flm_sess = flz.get_flexilims_session(project_id=camera_ds.project_id)
    target_name = f"{camera_ds.dataset_name}_deinterleaved"

    target_ds = flz.Dataset.from_origin(
        origin_id=camera_ds.origin_id,
        dataset_type=CameraData.DATASET_TYPE,
        base_name=target_name,
        conflicts=conflicts,
        flexilims_session=flm_sess,
    )

    if target_ds.flexilims_status() != "not online" and conflicts == "skip":
        print(f"Skipping {target_ds.full_name} as it already exists")
        return None, target_ds

    print("Deinterleaving %s" % camera_ds.full_name)

    slurm_folder = target_ds.path_full
    slurm_folder.mkdir(parents=True, exist_ok=True)
    job_id = deinterleave(
        camera_ds.id,
        project_id=camera_ds.project_id,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        job_dependency=dependency,
    )
    if not use_slurm:
        job_id = None

    return job_id, target_ds


@slurm_it(
    conda_env="cottage_analysis",
    slurm_options=dict(mem="8G", time="24:00:00"),
    module_list=["FFmpeg"],
    from_imports={"cottage_analysis.preprocessing.minicam": "deinterleave"},
)
def deinterleave(camera_ds_id, project_id):
    """Deinterleave a camera dataset.

    Will deinterleave the video file and update the dataset in flexilims.

    Args:
        camera_ds_id (str): id of the camera dataset
        project_id (str): id of the project
    """
    flm_sess = flz.get_flexilims_session(project_id=project_id)
    camera_ds = flz.Dataset.from_flexilims(id=camera_ds_id, flexilims_session=flm_sess)
    target_name = f"{camera_ds.dataset_name}_deinterleaved"
    target_ds = flz.Dataset.from_origin(
        origin_id=camera_ds.origin_id,
        dataset_type=CameraData.DATASET_TYPE,
        base_name=target_name,
        conflicts="skip",
        flexilims_session=flm_sess,
    )
    target_ds.extra_attributes = dict(
        metadata_file=camera_ds.extra_attributes["metadata_file"],
        timestamp_file=camera_ds.extra_attributes["timestamp_file"],
        video_file=target_name + ".mp4",
    )
    target_ds.path_full.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        camera_file=camera_ds.path_full / camera_ds.extra_attributes["video_file"],
        target_file=target_ds.path_full / target_ds.extra_attributes["video_file"],
        make_grey=False,
        verbose=True,
        intrinsic_calibration=None,
    )

    # copy timestamp and metadata files
    for file in ["timestamp_file", "metadata_file"]:
        raw = camera_ds.path_full / camera_ds.extra_attributes[file]
        if not raw.exists():
            print(f"Warning: {raw} does not exist")
            continue
        if file == "timestamp_file":
            # read timestamp file and double the number of rows, adding half a frame
            # in between each frame
            raw_df = pd.read_csv(raw, parse_dates=["BonsaiTimestamp"])
            deinterleave_df = pd.DataFrame(
                index=np.arange(raw_df.shape[0] * 2), columns=raw_df.columns
            )
            deinterleave_df["frame_id"] = np.arange(raw_df.shape[0] * 2)
            for timestamps in ["BonsaiTimestamp", "HarpTimestamp"]:
                raw_times = raw_df[timestamps].values
                if raw_times.dtype == "<M8[ns]":
                    convert = True
                    raw_times = raw_times.astype("long")
                else:
                    convert = False
                # frames are timestamped after production, so raw index starts at 1
                raw_index = np.arange(len(raw_times)) * 2 + 1
                deinterleaved_index = np.arange(len(raw_times) * 2)
                deinterleaved_times = np.interp(
                    deinterleaved_index,
                    raw_index,
                    raw_times,
                )
                if convert:
                    deinterleaved_times = pd.to_datetime(deinterleaved_times)

                assert all(raw_df[timestamps].values == deinterleaved_times[1::2])
                deinterleave_df[timestamps] = deinterleaved_times
                assert deinterleave_df[timestamps].is_monotonic_increasing

            deinterleave_df.to_csv(
                target_ds.path_full / target_ds.extra_attributes[file],
                index=False,
            )
            frame_rate = 1 / np.median(np.diff(deinterleave_df["HarpTimestamp"].values))
            kwargs["frame_rate"] = frame_rate
        else:
            # deinterleaving will reduce the frame height by 1
            with open(raw, "r") as f:
                d = yaml.safe_load(f)
            if "height" in d:
                d["height"] -= 1
            elif "Height" in d:
                d["Height"] -= 1
            else:
                raise ValueError("Could not find height in metadata file")
            with open(target_ds.path_full / target_ds.extra_attributes[file], "w") as f:
                yaml.dump(d, f)

    video.io_func.deinterleave_camera(**kwargs)

    target_ds.extra_attributes.update(kwargs)
    target_ds.update_flexilims(mode="overwrite")
    return target_ds.path_full
