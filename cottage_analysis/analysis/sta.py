import numpy as np
import pandas as pd
import pickle
from cottage_analysis.stimulus_structure import spheres_tube
from cottage_analysis.analysis import common_utils
from cottage_analysis.filepath import generate_filepaths


# Regenerate sphere stimuli for each recording
def regenerate_stimuli(project, mouse, session, protocol):
    def regenerate_stimuli_each_recording(
        project,
        mouse,
        session,
        protocols,
        protocol,
        irecording,
        nrecordings,
    ):
        (
            rawdata_folder,
            protocol_folder,
            _,
            _,
            _,
        ) = generate_filepaths.generate_file_folders(
            project=project,
            mouse=mouse,
            session=session,
            protocol=protocol,
            all_protocol_recording_entries=None,
            recording_no=0,
        )

        param_log = pd.read_csv(rawdata_folder / "NewParams.csv")
        param_log = param_log.rename(columns={"Radius": "Depth"})

        with open(protocol_folder / "sync/imaging_df.pickle", "rb") as handle:
            imaging_df = pickle.load(handle)
        with open(protocol_folder / "sync/vs_df.pickle", "rb") as handle:
            vs_df = pickle.load(handle)
        with open(protocol_folder / "sync/trials_df.pickle", "rb") as handle:
            trials_df = pickle.load(handle)
        output = spheres_tube.regenerate_frames(
            frame_times=imaging_df[
                "harptime_imaging_trigger"
            ].values,  # using imaging frames as the list of timepoints to reconstruct stimuli
            trials_df=trials_df,
            vs_df=vs_df,
            param_logger=param_log,
            time_column="HarpTime",
            resolution=1,
            sphere_size=10,
            azimuth_limits=(-120, 120),
            elevation_limits=(-40, 40),
            verbose=True,
            output_datatype="int16",
            output=None,
        )

        np.save(protocol_folder / "stimuli.npy", output)

        return output

    outputs = []
    output = common_utils.loop_through_recordings(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        func=regenerate_stimuli_each_recording,
    )
    outputs.append(output)
    outputs = np.stack(output)

    return output


def sta_by_depth(
    trials_df,
    reconstructed_frames,
    frame_times,
    frame_rate,
    delays=None,
    spk_per_frame=None,
    verbose=True,
):
    """Spike triggered average of reconstructed frames by depth

    Delays, in second, are delay applied to the stimulus sequence. If delay is -100,
    that means that spikes were triggered by stimulus 100ms before them.

    Args:
        trials_df (pd.DataFrame): stimulus structure with each row as a trial.
        reconstructed_frames (np.array): n frames x n elev x n azim binary array of
                                         stimuli
        frame_times (np.array): time of each frame, same unit as corridor_df.start_time
        frame_rate (float): frame rate to calculate delays. 144 for monitor frames, 15 for imaging frames.
        delays (np.array): array of delays in seconds
        spk_per_frame (np.array): spike for each frame, use to weight average. If None
                                  will do simple average
        verbose (bool): print progress

    Returns:
        sta (np.array): n depth x n delay x n elev x n azim weighted average
        nspkes (np.array): n depth vector of number of spikes
        depths (np.array): ordered depths corresponding to first sta dimension
        delays (np.array): ordered delays corresponding to second sta dimension
    """
    if delays is None:
        delays = [0]
    if spk_per_frame is None:
        spk_per_frame = np.ones(reconstructed_frames.shape[0])

    depths = np.sort(trials_df.depth.unique())
    full_sta = np.zeros((len(depths), len(delays), *reconstructed_frames.shape[1:]))
    nspks = np.zeros(len(depths))

    for idepth, depth in enumerate(depths):
        if verbose:
            print(f"... doing depth {depth*100} cm")
        depth_df = trials_df[trials_df.depth == depth]
        # find frames at this depth
        # starts = depth_df.imaging_frame_stim_start.values
        # ends = depth_df.imaging_frame_stim_stop.values
        starts = frame_times.searchsorted(depth_df.harptime_stim_start)
        ends = frame_times.searchsorted(depth_df.harptime_stim_stop)
        ends = ends[: len(starts)]
        frame_index = np.hstack(
            [np.arange(s, e, dtype=int) for s, e in zip(starts, ends)]
        )
        # keep non-shifted spikes for all delay
        # do it like that to look for valid frames only once
        spk_per_frame_at_depth = spk_per_frame[frame_index]
        nspks[idepth] = np.sum(spk_per_frame_at_depth)
        valid_frames = spk_per_frame_at_depth != 0
        for idelay, delay in enumerate(delays):
            if verbose:
                print(f"... ... doing delay {delay * 1000} ms")
            shift = int(delay * frame_rate)
            # shift the stim
            shifted_frames = np.clip(
                frame_index[valid_frames] + shift, 0, len(frame_times)
            )
            stims = reconstructed_frames[shifted_frames].reshape(
                len(shifted_frames), -1
            )
            sta = np.dot(stims.T, spk_per_frame_at_depth[valid_frames])
            sta = sta.reshape(reconstructed_frames.shape[1:])
            full_sta[idepth, idelay] = sta
            # !! Needs to add normalized STA
    return full_sta, nspks, depths, delays
