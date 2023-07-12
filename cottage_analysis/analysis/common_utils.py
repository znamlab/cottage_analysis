import numpy as np
import scipy
from scipy.optimize import curve_fit
import flexiznam as flz
from cottage_analysis.filepath import generate_filepaths
from cottage_analysis.preprocessing import synchronisation
from pathlib import Path


def get_confidence_interval(arr, sem=[], alpha=0.05, mean_arr=[]):
    """
    Get confidence interval of an input array using normal approximation.

    Args:
        arr (np.ndarray): 2d array, for example ndepths x ntrials to calculate
            confidence interval across trials.
        sem (np.ndarray): 1d array, for example ndepths to calculate confidence
            interval across depths.
        alpha (float, optional): Significant level. Default 0.05.
        mean_arr (np.ndarray): 1d array, for example ndepths to calculate confidence
            interval across depths.

    Returns:
        CI_low (np.1darray): lower bound of confidence interval.
        CI_high (np.1darray): upper bound of confidence interval.

    """
    z = scipy.stats.norm.ppf((1 - alpha / 2))
    if len(sem) > 0:
        sem = sem
    else:
        sem = scipy.stats.sem(arr, nan_policy="omit")
    if len(mean_arr) > 0:
        CI_low = mean_arr - z * sem
        CI_high = mean_arr + z * sem
    else:
        CI_low = np.average(arr, axis=0) - z * sem
        CI_high = np.average(arr, axis=0) + z * sem
    return CI_low, CI_high


def calculate_r_squared(y, y_hat):
    """Calculate R squared as the fraction of variance explained.

    Args:
        y: true values
        y_hat: predicted values

    """
    y = np.array(y)
    y_hat = np.array(y_hat)
    residual_var = np.sum((y_hat - y) ** 2)
    total_var = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - residual_var / total_var
    return r_squared


def iterate_fit(
    func, X, y, lower_bounds, upper_bounds, niter=5, p0_func=None, verbose=False
):
    """Iterate fitting to avoid local minima.

    Args:
        func: function to fit
        X: independent variables
        y: dependent variable
        lower_bounds: lower bounds for the parameters
        upper_bounds: upper bounds for the parameters
        niter: number of iterations
        p0_func: function to generate initial parameters

    Returns:
        popt_best: best parameters
        rsq_best: best R squared

    """
    popt_arr = []
    rsq_arr = []
    np.random.seed(42)
    for i_iter in range(niter):
        if p0_func is not None:
            p0 = p0_func()
        else:
            # generate random initial parameters from a standard normal distribution for unbounded parameters
            # otherwise, draw from a uniform distribution between lower and upper bounds
            p0 = np.random.normal(0, 1, len(lower_bounds))
            for i in range(len(lower_bounds)):
                if np.isinf(lower_bounds[i]) or np.isinf(upper_bounds[i]):
                    continue
                else:
                    p0[i] = np.random.uniform(lower_bounds[i], upper_bounds[i])
        popt, _ = curve_fit(
            func,
            X,
            y,
            maxfev=100000,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            p0=p0,
        )

        pred = func(np.array(X), *popt)
        r_sq = calculate_r_squared(y, pred)
        popt_arr.append(popt)
        rsq_arr.append(r_sq)
        if verbose:
            print(f"Iteration {i_iter}, R^2 = {r_sq}")
    idx_best = np.argmax(np.array(rsq_arr))
    popt_best = popt_arr[idx_best]
    rsq_best = rsq_arr[idx_best]
    return popt_best, rsq_best


def loop_through_recordings(project, mouse, session, protocol, func, **kwargs):
    """Help function to loop through recordings under protocols of the same session.

    Args:
        project (str): _description_
        mouse (_type_): _description_
        session (_type_): _description_
        protocol (_type_): _description_
        func (_type_): _description_
    """

    flexilims_session = flz.get_flexilims_session(project_id=project)
    sess_children = generate_filepaths.get_session_children(
        project=project,
        mouse=mouse,
        session=session,
        flexilims_session=flexilims_session,
    )
    if len(sess_children[sess_children.name.str.contains("Playback")]) > 0:
        protocols = [protocol, f"{protocol}Playback"]
    else:
        protocols = [protocol]

    for iprotocol, protocol in enumerate(protocols):
        print(
            f"---------Process protocol {iprotocol+1}/{len(protocols)}---------",
            flush=True,
        )

        flexilims_session = flz.get_flexilims_session(project_id=project)
        all_protocol_recording_entries = generate_filepaths.get_all_recording_entries(
            project,
            mouse,
            session,
            protocol=protocol,
            flexilims_session=flexilims_session,
        )

        nrecordings = len(all_protocol_recording_entries)
        # For each recording, synchronise and produce frames_df, trials_df
        for irecording in range(len(all_protocol_recording_entries)):
            return func(
                project=project,
                mouse=mouse,
                session=session,
                irecording=irecording,
                nrecordings=nrecordings,
                protocols=protocols,
                protocol=protocol,
                **kwargs,
            )


def concatenate_recordings(project, mouse, session, protocol="SpheresPermTubeReward"):
    """Concatenate vs_df and trials_df from multiple recordings under the same protocol.

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name
        protocol (str): protocol name of the closed loop experiment. Default = 'SpheresPermTubeReward'

    """
    # Make folder for this protocol (closedloop/playback)
    root = Path(flz.PARAMETERS["data_root"]["processed"])
    session_analysis_folder = root / project / mouse / session
    if not (session_analysis_folder / "plane0").exists():
        (session_analysis_folder / "plane0").mkdir(parents=True)

    flexilims_session = flz.get_flexilims_session(project_id=project)
    sess_children = generate_filepaths.get_session_children(
        project=project,
        mouse=mouse,
        session=session,
        flexilims_session=flexilims_session,
    )
    if len(sess_children[sess_children.name.str.contains("Playback")]) > 0:
        protocols = [protocol, f"{protocol}Playback"]
    else:
        protocols = [protocol]

    for iprotocol, protocol in enumerate(protocols):
        print(
            f"---------Process protocol {iprotocol+1}/{len(protocols)}---------",
            flush=True,
        )
        # ----- STEP1: Generate file path -----
        flexilims_session = flz.get_flexilims_session(project_id=project)
        all_protocol_recording_entries = generate_filepaths.get_all_recording_entries(
            project,
            mouse,
            session,
            protocol=protocol,
            flexilims_session=flexilims_session,
        )

        # For each recording, synchronise and produce frames_df, trials_df
        for irecording in range(len(all_protocol_recording_entries)):
            # synchronisation
            vs_df = synchronisation.generate_vs_df(
                project=project,
                mouse=mouse,
                session=session,
                protocol=protocol,
                irecording=irecording,
            )
            trials_df, imaging_df = synchronisation.generate_trials_df(
                project=project,
                mouse=mouse,
                session=session,
                protocol=protocol,
                vs_df=vs_df,
                irecording=irecording,
            )
            print(
                f"Synchronised recording {irecording+1}/{len(all_protocol_recording_entries)}",
                flush=True,
            )

            if (irecording == 0) and (protocol == protocols[0]):
                vs_df_all = vs_df.copy()
                vs_df_all["recording_no"] = irecording

                trials_df_all = trials_df.copy()
                trials_df_all["recording_no"] = irecording

                imaging_df_all = imaging_df.copy()
                imaging_df_all["recording_no"] = irecording
            else:
                vs_df_all = pd.read_pickle(
                    session_analysis_folder / "plane0/vs_df.pickle"
                )
                vs_df["recording_no"] = irecording
                vs_df_all = vs_df_all.append(vs_df, ignore_index=True)

                trials_df_all = pd.read_pickle(
                    session_analysis_folder / "plane0/trials_df.pickle"
                )
                imaging_df_all = pd.read_pickle(
                    session_analysis_folder / "plane0/imaging_df.pickle"
                )
                if protocol == protocols[0]:
                    is_closedloop = 1
                else:
                    is_closedloop = 0
                previous_trial_num = len(
                    trials_df_all[trials_df_all.closed_loop == is_closedloop]
                )
                trials_df["recording_no"] = irecording
                trials_df["trial_no"] = trials_df["trial_no"] + previous_trial_num
                imaging_df["recording_no"] = irecording
                trials_df_all = trials_df_all.append(trials_df, ignore_index=True)
                imaging_df_all = imaging_df_all.append(imaging_df, ignore_index=True)
            vs_df_all.to_pickle(session_analysis_folder / "plane0/vs_df.pickle")
            trials_df_all.to_pickle(session_analysis_folder / "plane0/trials_df.pickle")
            imaging_df_all.to_pickle(
                session_analysis_folder / "plane0/imaging_df.pickle"
            )

            print(
                f"Appended recording {irecording+1}/{len(all_protocol_recording_entries)}",
                flush=True,
            )


def load_is_cell_file(project, mouse, session, protocol="SpheresPermTubeReward"):
    (_, _, _, suite2p_path, _) = generate_filepaths.generate_file_folders(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        all_protocol_recording_entries=None,
        recording_no=0,
        flexilims_session=None,
    )
    iscell = np.load(suite2p_path / "iscell.npy", allow_pickle=True)[:, 0]
    iscell = iscell.astype("int")

    return iscell


def get_confidence_interval(arr, axis=1, sig_level=0.05):
    """Get confidence interval of an input array.

    Args:
        arr (np.ndarray): 2d array, for example ndepths x ntrials to calculate confidence interval across trials.
        sig_level (float, optional): Significant level. Default 0.05.

    Returns:
        CI_low (np.1darray): lower bound of confidence interval.
        CI_high (np.1darray): upper bound of confidence interval.
    """

    z = scipy.stats.norm.ppf((1 - sig_level / 2))
    sem = scipy.stats.sem(arr, axis=axis, nan_policy="omit")
    CI_low = np.average(arr, axis=axis) - z * sem
    CI_high = np.average(arr, axis=axis) + z * sem
    return CI_low, CI_high
