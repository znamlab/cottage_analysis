import os
from pathlib import Path
import flexiznam as flz
from . import slurm_job


def run_dlc(camera_ds, flm_sess, dlc_model, crop=False, redo=False):
    """Run dlc tracking on a camera dataset

    Wrapper around eye_tracking.slurm_job.slurm_dlc_pupil (which does the proper tracking)

    Will find the corresponding dlc dataset and delete it if redo is True.
    If crop is True, one uncropped dataset must exist, and a crop file will be created

    Args:
        camera_ds (Dataset): The camera dataset to track
        flm_sess (flexilims.Session): The flexilims session
        dlc_model (str): Name of the dlc model to use
        crop (bool, optional): Whether to crop the video. Defaults to False.
        redo (bool, optional): Whether to redo tracking if it already exists. If True,
            the tracking data and the flexilims entry will be deleted first, before
            rerunning the analysis. Defaults to False.
    """
    processed_path = Path(flz.PARAMETERS["data_root"]["processed"])
    dataset_path = Path(processed_path, camera_ds.path)

    # get flexilims entries
    dlc_datasets = flz.get_children(
        parent_id=camera_ds.origin_id,
        children_datatype="dataset",
        flexilims_session=flm_sess,
    )
    dlc_datasets = dlc_datasets[dlc_datasets["dataset_type"] == "dlc_tracking"]
    dlc_datasets = [
        flz.Dataset.from_flexilims(data_series=series, flexilims_session=flm_sess)
        for _, series in dlc_datasets.iterrows()
    ]

    ds_dict = dict(cropped=None, uncropped=None)
    for d in dlc_datasets:
        vid = d.extra_attributes["videos"]
        assert (
            len(vid) == 1
        ), f"{d.name} tracking with more than one video, is that normal?"
        # exclude tracking for other videos
        if not vid[0].endswith(camera_ds.extra_attributes["video_file"]):
            continue
        if isinstance(d.extra_attributes["cropping"], list):
            if ds_dict["cropped"] is not None:
                raise IOError("More than one cropped dataset")
            ds_dict["cropped"] = d
        else:
            if ds_dict["uncropped"] is not None:
                raise IOError("More than one uncropped dataset")
            ds_dict["uncropped"] = d

    which = "cropped" if crop else "uncropped"
    ds = ds_dict[which]
    if ds is not None:
        if not redo:
            print("  Already done. Skip")
            return
        else:
            print("  Erasing previous tracking to redo")
            # delete labeled and filtered version too
            filenames = []
            for suffix in ["", "_filtered"]:
                p = ds.path_full / ds.extra_attributes["dlc_file"]
                basename = p.with_name(p.stem + suffix)
                for ext in [".h5", ".csv"]:
                    filenames.append(basename.with_suffix(ext))
                filenames.append(basename.with_name(basename.stem + "_labeled.mp4"))
            filenames.append(p.with_name(p.stem + "_meta.pickle"))
            for fname in filenames:
                if fname.exists():
                    print(f"        deleting {fname}")
                    os.remove(fname)
            # also remove the flexilims entry
            flm_sess.delete(ds.id)

    if crop:
        if ds_dict["uncropped"] is None:
            print("No uncropped dataset found, skipping for now")
            return
        crop_info = slurm_job.create_crop_file(camera_ds, ds_dict["uncropped"])
        crop_info = [
            crop_info["xmin"],
            crop_info["xmax"],
            crop_info["ymin"],
            crop_info["ymax"],
        ]
    else:
        crop_info = None

    # Define the dataset name here, it will be used to upload on flexilims by dlc_track
    dataset_name = f"dlc_tracking_{camera_ds.dataset_name}_{which}"
    target_folder = dataset_path / dataset_name
    process = slurm_job.slurm_dlc_pupil(
        video_path=camera_ds.path_full / camera_ds.extra_attributes["video_file"],
        model_name=dlc_model,
        target_folder=target_folder,
        origin_id=camera_ds.origin_id,
        project=camera_ds.project,
        filter=False,
        label=False,
        crop_info=crop_info,
    )
    return process
