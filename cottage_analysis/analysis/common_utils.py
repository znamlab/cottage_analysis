import numpy as np
import scipy
import pandas as pd
import yaml
from tifffile import TiffFile
from scipy.optimize import curve_fit
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats

from cottage_analysis.analysis import find_depth_neurons, common_utils
import flexiznam as flz
from flexilims.offline import download_database


def bootstrap_sample(df, columns):
    """Hierarchically resample a dataframe for bootstrap.
    The resampling is done in a hierarchical manner, where the first column is resampled,
    then the second column is resampled for each value of the first column, and so on.
    After the last column, the indices are resampled for each combination of the previous columns.

    Args:
        df: dataframe
        columns: list of columns to resample

    Returns:
        resampled_values: indices of resampled rows

    """
    for i in range(len(columns) + 1):
        if i == 0:
            values = df[columns[i]].unique()
            resampled_values = np.random.choice(values, size=values.shape)
        else:
            all_values = []
            for prev_col in resampled_values:
                if i == len(columns):
                    values = df[df[columns[i - 1]] == prev_col].index.values
                else:
                    values = df[df[columns[i - 1]] == prev_col][columns[i]].unique()
                all_values.append(
                    np.random.choice(
                        values,
                        size=values.shape,
                    )
                )
            resampled_values = np.concatenate(all_values)
    return resampled_values


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
    valid = ~np.isnan(X) & ~np.isnan(y)
    if np.any(~valid):
        print(f"Warning: {np.sum(~valid)} NaN values in X or y")
        X = X[valid]
        y = y[valid]
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
            maxfev=100000,  # 100000
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


def get_confidence_interval(
    arr=(),
    mean_arr=(),
    sem_arr=(),
    axis=1,
    sig_level=0.05,
):
    """Get confidence interval of an input array.

    Args:
        arr (np.ndarray, optional): 2d array, for example ndepths x ntrials to calculate confidence interval across trials.
        mean_arr (np.ndarray, optional): mean array, if the original array is not provided.
        sem_arr (np.ndarray, optional): SEM array, if the original array is not provided.
        sig_level (float, optional): Significant level. Default 0.05.
        method (str): Method to calculate confidence interval, "normal" or "bootstrap". Default "normal".
    Returns:
        CI_low (np.1darray): lower bound of confidence interval.
        CI_high (np.1darray): upper bound of confidence interval.
    """

    z = scipy.stats.norm.ppf((1 - sig_level / 2))
    if len(arr) > 0:
        sem = scipy.stats.sem(arr, axis=axis, nan_policy="omit")
        CI_low = np.nanmean(arr, axis=axis) - z * sem
        CI_high = np.nanmean(arr, axis=axis) + z * sem
    elif len(mean_arr) > 0 and len(sem_arr) > 0:
        CI_low = mean_arr - z * sem
        CI_high = mean_arr + z * sem
    else:
        print("Error: you need to input either [arr] or [mean_arr and sem_arr]")
        CI_low = []
        CI_high = []
    return CI_low, CI_high


def get_bootstrap_ci(arr, sig_level=0.05, n_bootstraps=1000, func=np.nanmean):
    """Calculate confidence interval using bootstrap method.

    Args:
        arr (np.ndarray): 1d or 2d array, for example ndepths x ntrials to calculate confidence interval across trials.
        sig_level (float): Significant level.
        n_bootstraps (int): Number of bootstraps.

    Returns:
        CI_low (np.1darray): lower bound of confidence interval.
        CI_high (np.1darray): upper bound of confidence interval.
    """
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    ndepths, ntrials = arr.shape
    bootstrapped_means = np.zeros((n_bootstraps, ndepths))
    for i in range(n_bootstraps):
        for idepth in range(ndepths):
            resampled = arr[idepth, np.random.randint(0, ntrials, ntrials)]
            bootstrapped_means[i, idepth] = func(resampled)
    CI_low = np.percentile(bootstrapped_means, 100 * sig_level / 2, axis=0)
    CI_high = np.percentile(bootstrapped_means, 100 * (1 - sig_level / 2), axis=0)
    return CI_low, CI_high


def choose_trials_subset(trials_df, choose_trials, sfx="", by_depth=False):
    depth_list = find_depth_neurons.find_depth_list(trials_df)
    trial_number = len(trials_df) // len(depth_list)

    if choose_trials == None:  # fit all trials
        trials_df_chosen = trials_df
        sfx = ""
        choose_trial_nums = np.arange(trial_number)
    else:
        if choose_trials == "odd":  # fit odd trials
            trials_df_chosen = pd.DataFrame(columns=trials_df.columns)
            # choose odd trials from trials_df
            for depth in depth_list:
                trials_df_depth = trials_df[trials_df.depth == depth]
                trials_df_depth = trials_df_depth.iloc[::2, :]
                trials_df_chosen = pd.concat([trials_df_chosen, trials_df_depth])
            choose_trial_nums = np.arange(trial_number)[::2]
            sfx = "_crossval"
        if choose_trials == "even":  # fit even trials
            trials_df_chosen = pd.DataFrame(columns=trials_df.columns)
            # choose even trials from trials_df
            for depth in depth_list:
                trials_df_depth = trials_df[trials_df.depth == depth]
                trials_df_depth = trials_df_depth.iloc[1::2, :]
                trials_df_chosen = pd.concat([trials_df_chosen, trials_df_depth])
            choose_trial_nums = np.arange(trial_number)[1::2]
            sfx = "_crossval"
        if choose_trials is not None and isinstance(
            choose_trials, list
        ):  # if choose_trials is a given list
            if by_depth:
                trials_df_chosen = pd.DataFrame(columns=trials_df.columns)
                for depth in depth_list:
                    trials_df_depth = trials_df[trials_df.depth == depth]
                    trials_df_depth = trials_df_depth.iloc[choose_trials]
                    trials_df_chosen = pd.concat([trials_df_chosen, trials_df_depth])
                    sfx = sfx
            else:
                trials_df_chosen = trials_df.iloc[choose_trials]
                choose_trial_nums = choose_trials
                sfx = sfx
            choose_trial_nums = choose_trials

    return trials_df_chosen, choose_trial_nums, sfx


def find_thresh_sequence(
    array,
    length,
    shift,
    threshold_min=None,
    threshold_max=None,
    mode="all",
):
    """Find a sequance within an array where before the indices, either the values are all within a range for a certain length, or the average of the previous values are within a range.
       For example, if shift = length = 15, the index indicates the current frame and the previous 14 frames are within a certain range.

    Args:
        array (np.1darray): array to be searched
        length (int): length of sequence
        shift (int): length of shift to the right (positive) or left (negative); positive shift = the sequence is in the past
        threshold_min (float, optional): min threshold. Defaults to None.
        threshold_max (float, optional): max threshold. Defaults to None.
        mode (str, optional): "all" or "average". Defaults to "all". All: all values in the previous sequence are within the threshold. Average: the average of the values in the previous sequence is within the threshold.

    Returns:
        _type_: _description_
    """
    assert mode in ["all", "average"], "mode must be either 'all' or 'average'"
    if (threshold_min is None) and (threshold_max is None):
        print("WARNING: No threshold is given. Full array is returned.")
        indices = np.arange(len(array))
    else:
        if mode == "all":
            # shift an array by the shift amount
            if threshold_min is None:
                mask = array < threshold_max
            elif threshold_max is None:
                mask = array > threshold_min
            else:
                mask = (array > threshold_min) & (array < threshold_max)
            conv = np.convolve(mask, np.ones(length, dtype=int), "valid")
            indices = np.where(conv >= length)[0]
            indices = indices + int(shift) - 1

            # Get rid of the indices that's larger than the length of the array
            indices = indices[indices < len(array)]
        elif mode == "average":
            # Calculate the rolling average according to the (length-1) frames before and current frame
            rolling_avg = pd.Series(array).rolling(length).mean()

            # Find indices where the rolling average within the range
            if threshold_min is None:
                mask = rolling_avg < threshold_max
            elif threshold_max is None:
                mask = rolling_avg > threshold_min
            else:
                mask = (rolling_avg > threshold_min) & (rolling_avg < threshold_max)
            indices = np.where(mask == 1)[0] - int(length) + int(shift)

        return indices


def fill_missing_elements(arr, fill_n):
    # if an element is larger than the previous element by more than 1, insert the consecutive x more integers after the previous element in the array.
    diffs = np.diff(arr)
    gap_indices = np.where(diffs > 1)[0]

    # Generate the missing numbers for each gap
    if len(gap_indices) > 0:
        new_elems = [np.arange(arr[i] + 1, arr[i] + fill_n) for i in gap_indices]

        # Concatenate the original array with the new elements and flatten
        filled_array = np.sort(np.concatenate((arr, np.concatenate(new_elems))))

    else:
        filled_array = arr

    return filled_array


def ztest_2d(x, mu0=(0, 0)):
    """2D equivalent of the Z-test using the Mahalanobis distance comparing
    the mean of the input array to a given mean.

    Args:
        x (np.ndarray): input array
        mu0 (tuple, optional): mean to compare to. Defaults to (0, 0).

    Returns:
        z_values (np.ndarray): z values
        p_values (np.ndarray): p values

    """
    mu = (np.mean(x, axis=0) - mu0)[:, np.newaxis]
    # Mahalanobis distance squared between the mean of the input array and the given mean
    d2 = (mu.T @ np.linalg.inv(np.cov(x.T)) @ mu)[0, 0]
    # p-value computed using chi-squared distribution
    pval = np.exp(-d2 / 2)
    return pval, d2


def download_full_flexilims_database(flexilims_session, target_file=None):
    """Download the full flexilims database as json and save to file

    Args:
        flexilims_session (flz.Session): Flexilims session
        target_file (str, optional): Path to save json file. Defaults to None.

    Returns:
        dict: The json data
    """

    json_data = download_database(
        flexilims_session, root_datatypes=("mouse"), verbose=True
    )
    if target_file is not None:
        with open(target_file, "w") as f:
            yaml.dump(json_data, f)
    return json_data


def get_si_metadata(flexilims_session, session):
    recording = flz.get_children(
        parent_name=session,
        flexilims_session=flexilims_session,
        children_datatype="recording",
    ).iloc[0]
    dataset = flz.get_children(
        parent_name=recording["name"],
        flexilims_session=flexilims_session,
        children_datatype="dataset",
        filter={"dataset_type": "scanimage"},
    ).iloc[0]
    data_root = flz.get_data_root("raw", flexilims_session=flexilims_session)
    if (data_root / recording["path"] / sorted(dataset["tif_files"])[0]).exists():
        tif_path = data_root / recording["path"] / sorted(dataset["tif_files"])[0]
        metadata = TiffFile(tif_path).scanimage_metadata
    else:
        suite2p_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=recording.name,
            dataset_type="suite2p_traces",
            filter_datasets={"anatomical_only": 3},
            allow_multiple=False,
            return_dataseries=False,
        )
        metadata_path = suite2p_ds.path_full / "si_metadata.npy"
        metadata = np.load(metadata_path, allow_pickle=True).item()
    return metadata


def create_nested_nan_list(levels):
    nested_list = np.nan  # Start with np.nan
    for _ in range(levels):
        nested_list = [nested_list]  # Wrap the current structure in a new list
    return [nested_list]


def dict2df(dict, df, cols, index):
    for key, item in dict.items():
        if key in cols:
            if isinstance(item, float):
                df[key].iloc[index] = item
            elif isinstance(item, list):
                df[key] = create_nested_nan_list(1)
                df[key].iloc[index] = item
            elif isinstance(item, np.ndarray):
                df[key] = create_nested_nan_list(item.ndim)
                df[key].iloc[index] = item.tolist()
    return df


def find_columns_containing_string(df, substring):
    return [col for col in df.columns if substring in col]


def ceil(a, base=1, precision=1):
    fold = a // (base * (10 ** (-precision)))
    extra = int((a % (base * (10 ** (-precision)))) > 0)
    ceiled_num = (fold + extra) * (base * (10 ** (-precision)))
    return np.round(ceiled_num, precision)


def hierarchical_bootstrap_stats(
    data,
    n_boots,
    xcol,
    resample_cols,
    ycol=None,
    correlation=False,
    difference=False,
    ratio=False,
):
    np.random.seed(0)
    if "mouse" not in data.columns:
        data["mouse"] = data["session"].str.split("_").str[0]
    distribution = np.zeros((n_boots, len(xcol)))
    r = np.zeros(len(xcol))
    if correlation:
        for icol, (x, y) in enumerate(zip(xcol, ycol)):
            r[icol] = stats.spearmanr(data[x], data[y])[0]
    elif difference:
        for icol, (x, y) in enumerate(zip(xcol, ycol)):
            r[icol] = np.median(data[x] - data[y])
    elif ratio:
        for icol, (x, y) in enumerate(zip(xcol, ycol)):
            r[icol] = np.median(data[x] / data[y])
    else:
        r = None
    for i in tqdm(range(n_boots)):
        sample = common_utils.bootstrap_sample(data, resample_cols)
        if ycol is None:
            for icol, x in enumerate(xcol):
                distribution[i, icol] = np.median(data.loc[sample][x])
        else:
            for icol, (x, y) in enumerate(zip(xcol, ycol)):
                if correlation:
                    distribution[i, icol] = stats.spearmanr(
                        data.loc[sample][x], data.loc[sample][y]
                    )[0]
                if difference:
                    distribution[i, icol] = np.median(
                        data.loc[sample][x] - data.loc[sample][y]
                    )
                if ratio:
                    distribution[i, icol] = np.median(
                        data.loc[sample][x] / data.loc[sample][y]
                    )
    plt.figure()
    for icol, x in enumerate(xcol):
        plt.subplot(2, len(xcol) // 2 + 1, icol + 1)
        plt.hist(distribution[:, icol], bins=31)
        plt.axvline(
            np.percentile(distribution[:, icol], 2.5), color="r", linestyle="--"
        )
        plt.axvline(
            np.percentile(distribution[:, icol], 97.5), color="r", linestyle="--"
        )
    return r, distribution


def calculate_pval_from_bootstrap(distribution, value):
    distribution = np.array(distribution)
    q_min = np.min([np.mean(distribution > value), np.mean(distribution < value)])
    return q_min * 2
