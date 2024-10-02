from roifile import ImagejRoi
import matplotlib.path as mplPath
from sklearn.linear_model import HuberRegressor
import flexiznam as flz
from matplotlib import pyplot as plt
import numpy as np
from cottage_analysis.analysis import common_utils


def find_roi_centers(neurons_df, stat):
    for roi in neurons_df.roi:
        ypix = stat[roi]["ypix"][~stat[roi]["overlap"]]
        xpix = stat[roi]["xpix"][~stat[roi]["overlap"]]
        neurons_df.at[roi, "center_x"] = np.mean(xpix)
        neurons_df.at[roi, "center_y"] = np.mean(ypix)


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
    for roi in neurons_df.roi:
        ypix = stat[roi]["ypix"][~stat[roi]["overlap"]]
        xpix = stat[roi]["xpix"][~stat[roi]["overlap"]]
        neurons_df.at[roi, "center_x"] = np.mean(xpix)
        neurons_df.at[roi, "center_y"] = np.mean(ypix)


def determine_roi_locations(neurons_df, flexilims_session, session, suite2p_ds):
    stat = np.load(suite2p_ds.path_full / "plane0" / "stat.npy", allow_pickle=True)
    ops = np.load(suite2p_ds.path_full / "plane0" / "ops.npy", allow_pickle=True).item()
    si_metadata = common_utils.get_si_metadata(flexilims_session, session)
    if "FrameData" in si_metadata.keys():
        neurons_df["z_position"] = si_metadata["FrameData"]["SI.hMotors.samplePosition"][2]
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
