from roifile import ImagejRoi
import matplotlib.path as mplPath
from sklearn.linear_model import HuberRegressor
import flexiznam as flz
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from cottage_analysis.analysis import common_utils


def align_across_mice(neurons_df, ref_mouse="PZAH10.2d"):
    neurons_df["mouse"] = neurons_df["session"].str.split("_").str[0]
    sig_neurons = (
        (neurons_df["rf_sig"] == True)
        & (neurons_df["iscell"] == True)
        & (neurons_df["overview_x"].isna() == False)
        & (neurons_df["v1"] == True)
    )
    mice = neurons_df["mouse"].unique()
    # add one hot encoding for each mouse
    for mouse in mice:
        neurons_df[mouse] = neurons_df["mouse"] == mouse
    # make a predictor matrix that includes the one hot encoding for each mouse
    # and overview_x and overview_y
    X = neurons_df[sig_neurons][["rf_azi", "rf_ele"]].values
    X = np.hstack([X, neurons_df[sig_neurons][mice].values])
    for col in "overview_x", "overview_y":
        y = neurons_df[sig_neurons][col].values
        # use Huber regression to fit X vs y
        huber = HuberRegressor(max_iter=100000)
        huber.fit(X, y)
        # correct the overview_x and overview_y by subtracting the coefficients corresponding to the one hot encoding
        # for each mouse
        mouse_offset = huber.coef_[2:]
        ref_mouse_offset = mouse_offset[mice == ref_mouse]
        X_all = neurons_df[["rf_azi", "rf_ele"]].values
        X_all = np.hstack([X_all, neurons_df[mice].values])
        y_all = neurons_df[col].values
        y_corrected = y_all - np.dot(X_all[:, 2:], mouse_offset) + ref_mouse_offset
        neurons_df[f"{col}_aligned"] = y_corrected
    return neurons_df


def check_neurons_in_v1(
    neurons_df,
    v1_mask_fname="/camp/lab/znamenskiyp/home/shared/projects/hey2_3d-vision_foodres_20220101/PZAH10.2d/FOVs/V1_mask_2.roi",
    overview_fname="/camp/lab/znamenskiyp/home/shared/projects/hey2_3d-vision_foodres_20220101/PZAH10.2d/FOVs/PZAH10.2d_overview.tif",
):
    v1_mask = ImagejRoi.fromfile(v1_mask_fname)
    overview_img = plt.imread(overview_fname)
    # make a boolean mask from polygon defined by v1_mask.coordinates()
    v1_mask_img = np.zeros(overview_img.shape[:2], dtype=bool)
    # Create a Path object from the vertices
    poly_path = mplPath.Path(v1_mask.coordinates())
    # Create a meshgrid for the image size
    y, x = np.mgrid[: overview_img.shape[0], : overview_img.shape[1]]
    # Create a binary mask by checking if each point in the image is within the polygon
    v1_mask_img = poly_path.contains_points(
        np.vstack((x.flatten(), y.flatten())).T
    ).reshape(x.shape)
    v1_mask_img = np.fliplr(v1_mask_img)

    def inside_mask(row):
        outside_mask_img = (
            (row["overview_x_aligned"] < 0)
            | (row["overview_x_aligned"] >= v1_mask_img.shape[1])
            | (row["overview_y_aligned"] < 0)
            | (row["overview_y_aligned"] >= v1_mask_img.shape[0])
        )
        if outside_mask_img or np.isnan(row["overview_x_aligned"]):
            return np.nan
        else:
            return v1_mask_img[
                int(row["overview_y_aligned"]), int(row["overview_x_aligned"])
            ]

    neurons_df["v1_mask"] = neurons_df.apply(inside_mask, axis=1)
    return overview_img


def load_overview_roi(flexilims_session, session):
    session_path = flz.get_path(
        session, flexilims_session=flexilims_session, datatype="session"
    )
    data_root = flz.get_data_root("processed", flexilims_session=flexilims_session)

    fovs = (data_root / session_path).parent / "FOVs" / "rois.zip"
    try:
        rois = ImagejRoi.fromfile(fovs)
    except FileNotFoundError:
        print("No overview ROI file found for session", session)
        return None
    session_date = session.split("_")[1]
    for roi in rois:
        if roi.name == session_date:
            return [roi.top, roi.bottom, roi.left, roi.right]
    print("No overview ROI found for session", session)
    return None


def find_roi_centers(neurons_df, stat):
    """Add the centre of each ROI, in FOV pixels, to neurons_df in place.

    The centre is the mean position of the ROI pixels that do not overlap other
    ROIs. ROIs that are NaN or not in `stat` are skipped and keep their existing
    centre, NaN if the columns did not exist.

    Args:
        neurons_df (pd.DataFrame): dataframe with a `roi` column of suite2p ROI
            indices. `center_x` and `center_y` columns are added if missing.
        stat (np.ndarray): suite2p `stat.npy` array, one dict per ROI.
    """
    if "center_x" not in neurons_df.columns:
        neurons_df["center_x"] = np.nan
    if "center_y" not in neurons_df.columns:
        neurons_df["center_y"] = np.nan
    for idx, roi in zip(neurons_df.index, neurons_df.roi):
        if pd.isna(roi):
            continue
        roi_int = int(roi)
        if roi_int < 0 or roi_int >= len(stat):
            continue
        ypix = stat[roi_int]["ypix"][~stat[roi_int]["overlap"]]
        xpix = stat[roi_int]["xpix"][~stat[roi_int]["overlap"]]
        neurons_df.loc[idx, "center_x"] = np.mean(xpix)
        neurons_df.loc[idx, "center_y"] = np.mean(ypix)


def determine_roi_locations(
    neurons_df, flexilims_session, session, suite2p_ds, filter_datasets
):
    stat = np.load(suite2p_ds.path_full / "plane0" / "stat.npy", allow_pickle=True)
    ops = np.load(suite2p_ds.path_full / "plane0" / "ops.npy", allow_pickle=True).item()
    si_metadata = common_utils.get_si_metadata(
        flexilims_session, session, filter_datasets
    )
    if "FrameData" in si_metadata.keys():
        neurons_df["z_position"] = si_metadata["FrameData"][
            "SI.hMotors.samplePosition"
        ][2]
    else:
        neurons_df["z_position"] = si_metadata["SI.hMotors.samplePosition"][2]
    find_roi_centers(neurons_df, stat)
    fov = load_overview_roi(flexilims_session, session)
    if fov is not None:
        neurons_df["fov"] = neurons_df["roi"].apply(lambda x: fov)
        fov_width = fov[3] - fov[2]
        fov_height = fov[1] - fov[0]
        neurons_df["overview_x"] = (
            neurons_df["center_x"] / ops["Lx"] * fov_width + fov[2]
        )
        neurons_df["overview_y"] = (
            neurons_df["center_y"] / ops["Ly"] * fov_height + fov[0]
        )


def spatial_gradient(x, y, values, n_perm=10000, seed=0):
    """Fit a linear gradient of `values` over 2D position and test its significance.

    Fits values ~ x + y jointly by least squares. Significance comes from the OLS
    F-test and from a permutation test that shuffles values across positions.
    Both assume independent observations, so for neighbouring cells, which are
    spatially correlated, the p-values are optimistic.

    Args:
        x (np.ndarray): x position of each observation, e.g. neurons_df.center_x.
        y (np.ndarray): y position of each observation, e.g. neurons_df.center_y.
        values (np.ndarray): value at each position. Non-finite values are dropped.
        n_perm (int): number of permutations. Defaults to 10000.
        seed (int): seed for the permutations. Defaults to 0.

    Returns:
        dict: n (observations used), slope_x and slope_y (value units per position
            unit), magnitude (norm of the slopes), direction (degrees,
            atan2(slope_y, slope_x)), r2, pval_f (F-test) and pval_perm.
    """
    x, y, values = (np.asarray(a, dtype=float) for a in (x, y, values))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    X = np.column_stack([np.ones(ok.sum()), x[ok], y[ok]])
    v = values[ok]
    n = len(v)
    rng = np.random.default_rng(seed)
    # Column 0 is the data, the others its permutations, so one lstsq fits them all
    V = np.column_stack([v] + [rng.permutation(v) for _ in range(n_perm)])
    coefs = np.linalg.lstsq(X, V, rcond=None)[0]
    resid = V - X @ coefs
    r2 = 1 - np.sum(resid**2, axis=0) / np.sum((v - v.mean()) ** 2)
    f_stat = (r2[0] / 2) / ((1 - r2[0]) / (n - 3))
    slope_x, slope_y = coefs[1:, 0]
    return dict(
        n=n,
        slope_x=slope_x,
        slope_y=slope_y,
        magnitude=np.hypot(slope_x, slope_y),
        direction=np.degrees(np.arctan2(slope_y, slope_x)),
        r2=r2[0],
        pval_f=scipy.stats.f.sf(f_stat, 2, n - 3),
        pval_perm=(np.sum(r2[1:] >= r2[0]) + 1) / (n_perm + 1),
    )
