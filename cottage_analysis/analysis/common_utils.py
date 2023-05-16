import numpy as np
from scipy.optimize import curve_fit
import flexiznam as flz
from cottage_analysis.filepath import generate_filepaths


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


def iterate_fit(func, X, y, lower_bounds, upper_bounds, niter=5, p0_func=None):
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
    for _ in range(niter):
        if p0_func is not None:
            p0 = p0_func()
        else:
            p0 = None
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
