import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from functools import partial
import flexiznam as flz
from cottage_analysis.eye_tracking import slurm_job
from cottage_analysis.eye_tracking import eye_model_fitting


def run_dlc(
    camera_ds,
    flexilims_session,
    dlc_model,
    crop=False,
    redo=False,
    use_slurm=True,
    slurm_folder=None,
    job_dependency=None,
):
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
        use_slurm (bool, optional): Whether to use slurm to run the job. Defaults to
            True.
        slurm_folder (str, optional): Path to the folder where to create the slurm scripts
            and slurm logs. Defaults to None, in which case the folder will be created
            using from_flexilims.
        job_dependency (str, optional): Job id to wait for before starting the job.
            Defaults to None.
    """

    # Check if analysis is already done and delete output if redo=True
    ds_dict = get_tracking_datasets(camera_ds, flexilims_session)
    which = "cropped" if crop else "uncropped"
    ds = ds_dict[which]
    if ds is not None:
        if not redo:
            print("  Already done. Skip")
            return
        else:
            print("  Erasing previous tracking to redo")
            # delete labeled and filtered version too. DLC would just not do anything
            # if the output files already exist
            delete_tracking_dataset(ds, flexilims_session)

    # Now start the job/function
    if use_slurm:
        func = partial(
            slurm_job.slurm_dlc_pupil,
            slurm_folder=slurm_folder,
            job_dependency=job_dependency,
        )
    else:
        func = dlc_pupil

    process = func(
        camera_ds_id=camera_ds.id,
        model_name=dlc_model,
        origin_id=camera_ds.origin_id,
        project=camera_ds.project,
        crop=crop,
    )
    return process


def delete_tracking_dataset(ds, flexilims_session):
    """Delete a dlc_tracking dataset

    Args:
        ds (flexiznam.Dataset): dlc_tracking dataset to delete
        flexilims_session (flexilims.Session): Flexilims session
    """
    filenames = []
    for suffix in ["", "_filtered"]:
        p = ds.path_full / ds.extra_attributes["dlc_file"]
        basename = p.with_name(p.stem + suffix)
        for ext in [".h5", ".csv"]:
            filenames.append(basename.with_suffix(ext))
        filenames.append(basename.with_name(basename.stem + "_labeled.mp4"))
    filenames.append(p.with_name(p.stem + "_meta.pickle"))
    # also delete slurm files
    for ext in ["sh", "py", "err", "out"]:
        filenames.append(ds.path_full / f"dlc_track.{ext}")
    for fname in filenames:
        if fname.exists():
            print(f"        deleting {fname}")
            os.remove(fname)
    # also remove the flexilims entry
    flexilims_session.delete(ds.id)


def dlc_pupil(
    camera_ds_id,
    model_name,
    origin_id,
    project,
    crop=False,
    conflicts="abort",
):
    """Run dlc tracking on a video

    This is the function that actually runs the tracking. It is called by run_dlc
    directly when not using slurm or by slurm_job.slurm_dlc_pupil when using slurm.

    Args:
        video_path (str): Path to the video file
        model_name (str): Name of the dlc model to use. Must be in the `DLC_models`
        origin_id (str): hex code of the origin on flexilims
        project (str): Name of the project on flexilims
        crop (bool, optional): Whether to crop the video. Defaults to False.
        conflicts (str, optional): How to handle conflicts when creating the dataset on
            flexilims. Defaults to "abort".
    """
    # import dlc only in functions that need it as it takes a long time to load
    import deeplabcut

    flexilims_session = flz.get_flexilims_session(project)
    camera_ds = flz.Dataset.from_flexilims(
        flexilims_session=flexilims_session, id=camera_ds_id
    )
    ds_dict = get_tracking_datasets(camera_ds, flexilims_session)
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]

    processed_path = Path(flz.PARAMETERS["data_root"]["processed"])
    dlc_model_config = processed_path / "DLC_models" / model_name / "config.yaml"

    video_path = Path(video_path)

    if crop:
        uncropped_ds = ds_dict["uncropped"]
        assert uncropped_ds is not None, "No uncropped dataset found"
        crop_info = create_crop_file(camera_ds, uncropped_ds)
        crop_info = [
            crop_info["xmin"],
            crop_info["xmax"],
            crop_info["ymin"],
            crop_info["ymax"],
        ]
        suffix = "cropped"
    else:
        crop_info = None
        suffix = "uncropped"

    basename = f"{video_path.stem}_dlc_tracking_{suffix}"
    flm_sess = flz.get_flexilims_session(project)
    ds = flz.Dataset.from_origin(
        origin_id=origin_id,
        dataset_type="dlc_tracking",
        flexilims_session=flm_sess,
        base_name=basename,
        conflicts=conflicts,
    )
    target_folder = Path(ds.path_full)
    target_folder.mkdir(exist_ok=True, parents=True)

    print(f"Doing %s" % video_path, flush=True)
    analyse_kwargs = dict(
        config=dlc_model_config,
        videos=[str(video_path)],
        videotype="",
        shuffle=1,
        trainingsetindex=0,
        gputouse=None,
        save_as_csv=False,
        in_random_order=True,
        destfolder=str(target_folder),
        batchsize=None,
        cropping=crop_info,
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

    dlc_output = target_folder / f"{video_path.stem}{out}.h5"
    if not dlc_output.exists():
        raise IOError(
            f"DLC ran but I cannot find the output. {dlc_output} does not exist."
        )

    # Adding to flexilims
    print("Updating flexilims", flush=True)
    ds.extra_attributes = dict(
        analyse_kwargs,
        dlc_prefix=out,
        dlc_file=f"{video_path.stem}{out}.h5",
    )
    ds.update_flexilims(mode="overwrite")


def get_tracking_datasets(camera_ds, flexilims_session):
    """Get the dlc tracking datasets corresponding to a camera dataset

    This will raise an error if more than one dataset is found for a given type

    Args:
        camera_ds (flexilims.Dataset): Camera dataset
        flexilims_session (flexilims.Session): Flexilims session

    Returns:
        dict: Dictionary with keys "cropped" and "uncropped", containing the
            corresponding datasets. If no dataset is found, the corresponding value is
            None
    """
    dlc_datasets = flz.get_children(
        parent_id=camera_ds.origin_id,
        children_datatype="dataset",
        flexilims_session=flexilims_session,
    )
    dlc_datasets = dlc_datasets[dlc_datasets["dataset_type"] == "dlc_tracking"]
    ds_dict = dict(cropped=None, uncropped=None)
    for ds_name, series in dlc_datasets.iterrows():
        ds = flz.Dataset.from_flexilims(
            data_series=series, flexilims_session=flexilims_session
        )
        vid = ds.extra_attributes["videos"]
        assert (
            len(vid) == 1
        ), f"{ds_name} tracking with more than one video, is that normal?"
        # exclude tracking for other videos
        if not vid[0].endswith(camera_ds.extra_attributes["video_file"]):
            continue
        if isinstance(ds.extra_attributes["cropping"], list):
            if ds_dict["cropped"] is not None:
                raise IOError("More than one cropped dataset")
            ds_dict["cropped"] = ds
        else:
            if ds_dict["uncropped"] is not None:
                raise IOError("More than one uncropped dataset")
            ds_dict["uncropped"] = ds
    return ds_dict


def create_crop_file(camera_ds, dlc_ds):
    """Create a crop file for DLC tracking

    Uses the results of the uncropped tracking to find the crop area and save it in a
    crop file. This crop file can then be used to crop the video before tracking.

    Args:
        camera_ds (flexilims.Dataset): Camera dataset, must contain project information
        dlc_ds (flexilims.Dataset): dlc_tracking dataset, containing uncropped tracking

    Returns:
        dict: Crop information

    """
    if dlc_ds.project is None:
        raise IOError("dlc_tracking dataset has no project information")

    metadata_path = camera_ds.path_full / camera_ds.extra_attributes["metadata_file"]
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    crop_file = dlc_ds.path_full / f"{video_path.stem}_crop_tracking.yml"

    if crop_file.exists():
        print("Crop file already exists. Delete manually to redo")
        with open(crop_file, "r") as fhandle:
            crop_info = yaml.safe_load(fhandle)
        return crop_info

    with open(metadata_path, "r") as fhandle:
        metadata = yaml.safe_load(fhandle)
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    print("Creating crop file")
    dlc_res = pd.read_hdf(dlc_file)
    # Find DLC crop area
    borders = np.zeros((4, 2))
    for iw, w in enumerate(
        (
            "left_eye_corner",
            "right_eye_corner",
            "top_eye_lid",
            "bottom_eye_lid",
        )
    ):
        vals = dlc_res.xs(w, level=1, axis=1)
        vals.columns = vals.columns.droplevel("scorer")
        v = np.nanmedian(vals[["x", "y"]].values, axis=0)
        borders[iw, :] = v

    borders = np.vstack([np.nanmin(borders, axis=0), np.nanmax(borders, axis=0)])
    borders += ((np.diff(borders, axis=0) * 0.2).T @ np.array([[-1, 1]])).T
    for i, w in enumerate(["Width", "Height"]):
        borders[:, i] = np.clip(borders[:, i], 0, metadata[w])
    borders = borders.astype(int)
    crop_info = dict(
        xmin=int(borders[0, 0]),
        xmax=int(borders[1, 0]),
        ymin=int(borders[0, 1]),
        ymax=int(borders[1, 1]),
        dlc_source=str(dlc_ds.path),
        dlc_ds_id=dlc_ds.id,
    )
    with open(crop_file, "w") as fhandle:
        yaml.dump(crop_info, fhandle)
    print("Crop file created")
    return crop_info


def run_fit_ellipse(
    camera_ds,
    flexilims_session,
    likelihood_threshold=None,
    job_dependency=None,
    redo=False,
    use_slurm=True,
    slurms_folder=None,
):
    ds_dict = get_tracking_datasets(camera_ds, flexilims_session)
    if ds_dict["cropped"] is None:
        raise IOError("No cropped dataset found")
    dlc_ds = ds_dict["cropped"]
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    target = dlc_ds.path_full / f"{dlc_file.stem}_ellipse_fits.csv"
    if target.exists():
        if not redo:
            print("  Already done. Skip")
            return
        os.remove(target)
    if use_slurm:
        if slurms_folder is None:
            slurms_folder = dlc_ds.path_full
        func = partial(
            slurm_job.fit_ellipses,
            job_dependency=job_dependency,
            slurm_folder=slurms_folder,
        )
    else:
        func = fit_ellipse
    job_id = func(
        camera_ds_id=camera_ds.id,
        project_id=camera_ds.project_id,
        likelihood_threshold=likelihood_threshold,
    )
    return job_id


def fit_ellipse(
    camera_ds_id,
    project_id,
    likelihood_threshold=None,
):
    flexilims_session = flz.get_flexilims_session(project_id)
    camera_ds = flz.Dataset.from_flexilims(
        id=camera_ds_id, flexilims_session=flexilims_session
    )
    ds_dict = get_tracking_datasets(camera_ds, flexilims_session)
    if ds_dict["cropped"] is None:
        raise IOError("No cropped dataset found")
    dlc_ds = ds_dict["cropped"]
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    target = dlc_ds.path_full / f"{dlc_file.stem}_ellipse_fits.csv"

    assert dlc_file.exists()

    print(f"Doing %s" % dlc_file)
    ellipse_fits = eye_model_fitting.fit_ellipses(
        dlc_file,
        likelihood_threshold=likelihood_threshold,
    )
    print(f"Fitted, save to {target}")
    ellipse_fits.to_csv(target, index=False)
    print("Done")
