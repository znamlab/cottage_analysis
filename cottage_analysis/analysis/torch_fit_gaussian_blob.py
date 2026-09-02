from typing import Optional
from cottage_analysis.analysis.fit_gaussian_blob import (
    Gaussian2DParams,
    GaussianAdditiveParams,
    GaussianMultiplicativeParams,
    Gaussian1DParams,
    depth_class_labels,
    initial_fit_conditions,
)
from dataclasses import dataclass
from typing import Callable

import functools
import numpy as np
import pandas as pd

from cottage_analysis.analysis import torch_utils, common_utils
from cottage_analysis.analysis.torch_utils import (
    Gaussian1DBounds,
    Gaussian2DBounds,
    Gaussian2DCholeskyBounds,
    Gaussian2DAngleBounds,
    GaussianAdditiveBounds,
)

from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

import warnings

try:
    import torch
except Exception as exc:
    raise ImportError(
        "pytorch_gaussian_fits requires PyTorch. Install with: pip install torch"
    ) from exc


## Class for specifying model parameters and bounds
@dataclass
class AdamWFitConfig:
    lr: float = 0.05
    weight_decay: float = 1e-6
    n_steps: int = 1000
    loss_fn: str = "mse"
    smooth_l1_beta: float = 1.0


@dataclass
class CurveFitConfig:
    method: str = "trf"
    n_iters: int = 2000


## Core model functions to fit
def gaussian_1d(
    x: torch.Tensor,
    params: torch.Tensor,
    bounds: Gaussian1DBounds | None = None,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
) -> torch.Tensor:
    """Evaluate 1D Gaussian for all ROIs in parallel.

    Args:
        x: Tensor of shape (n_samples,) with log-RS or log-OF values.
        raw_params: Tensor of shape (n_rois, 4), unconstrained trainable params.
        bounds: Bounds for center parameter.
        min_sigma: Small additive term for variance stability.

    Returns:
        Tensor of shape (n_samples, n_rois).
    """
    # Unpack unconstrained params
    single = params.ndim == 1
    if single:
        params = params.unsqueeze(0)

    log_amplitude = params[:, 0]
    x0 = params[:, 1]
    log_sigma_x2 = params[:, 2]
    offset = params[:, 3]

    # Constrain selected parameters to match original fit semantics
    if optimiser is not None and optimiser.lower() == "adamw" and bounds is not None:
        x0 = torch_utils.bounded_sigmoid(x0, bounds.x0_min, bounds.x0_max)

    sigma_x_sq = torch.exp(log_sigma_x2) + min_sigma
    amplitude = torch.exp(log_amplitude)

    x_col = x[:, None]
    x_shift = x_col - x0[None, :]

    exponent = -(x_shift**2) / (2.0 * sigma_x_sq[None, :])
    result = offset[None, :] + amplitude[None, :] * torch.exp(exponent)
    return result.squeeze(-1) if single else result


def gaussian_bivar_cholesky(
    x: torch.Tensor,
    y: torch.Tensor,
    params: torch.Tensor,
    bounds: Gaussian2DCholeskyBounds,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
) -> torch.Tensor:
    """Cholesky-parameterised bivariate Gaussian.

    Args:
        x: Tensor of shape (n_samples,) with log-RS values.
        y: Tensor of shape (n_samples,) with log-OF values.
        params: Tensor of shape (n_rois, 7), unconstrained trainable params.
        bounds: Bounds for center and angle parameters.
        min_sigma: Small additive term for variance stability.

    Returns:
        Tensor of shape (n_samples, n_rois).
    """
    # Unpack unconstrained params
    single = params.ndim == 1
    if single:
        params = params.unsqueeze(0)

    log_amplitude = params[:, 0]
    x0 = params[:, 1]
    y0 = params[:, 2]
    log_l11 = params[:, 3]
    l21 = params[:, 4]
    log_l22 = params[:, 5]
    offset = params[:, 6]

    if optimiser is not None and optimiser.lower() == "adamw" and bounds is not None:
        log_amplitude = torch_utils.bounded_softplus_upper(
            log_amplitude, bounds.log_amplitude_max
        )
        x0 = torch_utils.bounded_sigmoid(x0, bounds.x0_min, bounds.x0_max)
        y0 = torch_utils.bounded_sigmoid(y0, bounds.y0_min, bounds.y0_max)
        log_l11 = torch_utils.bounded_sigmoid(
            log_l11, -bounds.log_l_max, bounds.log_l_max
        )
        log_l22 = torch_utils.bounded_sigmoid(
            log_l22, -bounds.log_l_max, bounds.log_l_max
        )
        l21 = torch_utils.bounded_sigmoid(l21, -bounds.l21_max, bounds.l21_max)

    amplitude = torch.exp(log_amplitude)
    basis = _g2d_basis(x, y, x0, y0, log_l11, l21, log_l22, min_sigma=min_sigma)
    result = offset[None, :] + amplitude[None, :] * basis

    return result.squeeze(-1) if single else result


def _g2d_basis(
    x: torch.Tensor,
    y: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    log_l11: torch.Tensor,
    l21: torch.Tensor,
    log_l22: torch.Tensor,
    min_sigma: float = 0.25,
) -> torch.Tensor:
    """Blob shape term of `gaussian_bivar_cholesky`

    Args:
        x, y: Tensors of shape (n_samples,).
        x0, y0, log_l11, l21, log_l22: Tensors of shape (n_candidates,)
        min_sigma: Small additive term for variance stability.

    Returns:
        Tensor of shape (n_samples, n_candidates).
    """
    a, b, c = torch_utils.regularised_precision(
        log_l11, l21, log_l22, min_sigma=min_sigma
    )
    delta_x = x[:, None] - x0[None, :]
    delta_y = y[:, None] - y0[None, :]

    exponent = -(
        a[None, :] * delta_x**2
        + 2.0 * b[None, :] * delta_x * delta_y
        + c[None, :] * delta_y**2
    )
    return torch.exp(exponent)


def gaussian_bivar_angle(
    x: torch.Tensor,
    y: torch.Tensor,
    params: torch.Tensor,
    bounds: Gaussian2DAngleBounds,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
) -> torch.Tensor:
    """Angle/sigma-parameterised bivariate Gaussian.

    Args:
        x: Tensor of shape (n_samples,) with log-RS values.
        y: Tensor of shape (n_samples,) with log-OF values.
        params: Tensor of shape (n_rois, 7), unconstrained trainable params.
        bounds: Bounds for center and angle parameters.
        min_sigma: Small additive term for variance stability.

    Returns:
        Tensor of shape (n_samples, n_rois).
    """
    # Unpack unconstrained params
    single = params.ndim == 1
    if single:
        params = params.unsqueeze(0)

    log_amplitude = params[:, 0]
    x0 = params[:, 1]
    y0 = params[:, 2]
    log_sigma_x2 = params[:, 3]
    log_sigma_y2 = params[:, 4]
    theta = params[:, 5]
    offset = params[:, 6]

    if optimiser is not None and optimiser.lower() == "adamw" and bounds is not None:
        # Constrain selected parameters to match original fit semantics
        x0 = torch_utils.bounded_sigmoid(x0, bounds.x0_min, bounds.x0_max)
        y0 = torch_utils.bounded_sigmoid(y0, bounds.y0_min, bounds.y0_max)
        theta = torch_utils.bounded_sigmoid(theta, bounds.theta_min, bounds.theta_max)

    sigma_x_sq = torch.exp(log_sigma_x2) + min_sigma
    sigma_y_sq = torch.exp(log_sigma_y2) + min_sigma
    amplitude = torch.exp(log_amplitude)

    delta_x = x[:, None] - x0[None, :]
    delta_y = y[:, None] - y0[None, :]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    a = cos_t**2 / (2.0 * sigma_x_sq) + sin_t**2 / (2.0 * sigma_y_sq)
    b = torch.sin(2.0 * theta) / (4.0 * sigma_x_sq) - torch.sin(2.0 * theta) / (
        4.0 * sigma_y_sq
    )
    c = sin_t**2 / (2.0 * sigma_x_sq) + cos_t**2 / (2.0 * sigma_y_sq)

    exponent = -(
        a[None, :] * delta_x**2
        + 2.0 * b[None, :] * delta_x * delta_y
        + c[None, :] * delta_y**2
    )
    result = offset[None, :] + amplitude[None, :] * torch.exp(exponent)

    return result.squeeze(-1) if single else result


def gaussian_2mult(
    x: torch.Tensor,
    y: torch.Tensor,
    raw_params: torch.Tensor,
    bounds: Gaussian2DBounds,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
) -> torch.Tensor:
    """Evaluate multiplicative 2D Gaussian for all ROIs in parallel.

    Args:
        x: Tensor of shape (n_samples,)
        y: Tensor of shape (n_samples,)
        raw_params: Tensor of shape (n_rois, 6), unconstrained trainable params.
        bounds: Bounds for center and angle parameters.
        min_sigma: Small additive term for variance stability.

    Returns:
        Tensor of shape (n_samples, n_rois).
    """
    # Unpack unconstrained params
    single = raw_params.ndim == 1
    if single:
        raw_params = raw_params.unsqueeze(0)

    log_amplitude = raw_params[:, 0]
    x0 = raw_params[:, 1]
    y0 = raw_params[:, 2]
    log_sigma_x2 = raw_params[:, 3]
    log_sigma_y2 = raw_params[:, 4]
    offset = raw_params[:, 5]

    # Constrain selected parameters to match original fit semantics
    if optimiser is not None and optimiser.lower() == "adamw" and bounds is not None:
        x0 = torch_utils.bounded_sigmoid(x0, bounds.x0_min, bounds.x0_max)
        y0 = torch_utils.bounded_sigmoid(y0, bounds.y0_min, bounds.y0_max)

    sigma_x_sq = torch.exp(log_sigma_x2) + min_sigma
    sigma_y_sq = torch.exp(log_sigma_y2) + min_sigma
    amplitude = torch.exp(log_amplitude)

    x_vec = x[:, None]
    y_vec = y[:, None]
    delta_x = x_vec - x0[None, :]
    delta_y = y_vec - y0[None, :]

    exponent_x = -(delta_x**2) / (2.0 * sigma_x_sq[None, :])
    exponent_y = -(delta_y**2) / (2.0 * sigma_y_sq[None, :])

    result = offset[None, :] + amplitude[None, :] * torch.exp(exponent_x) * torch.exp(
        exponent_y
    )
    return result.squeeze(-1) if single else result


## Wrapper functions for each model variant that wrap around the basic gaussian_1d or gaussian_2d functions
def gaussian_of(
    xy_tuple: tuple[torch.Tensor, torch.Tensor],
    params: torch.Tensor,
    bounds: Gaussian1DBounds,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
):
    """Evaluate 1D Gaussian for optic flow."""
    rs, of = xy_tuple
    return gaussian_1d(of, params, bounds, min_sigma, optimiser=optimiser)


def gaussian_rs(
    xy_tuple: tuple[torch.Tensor, torch.Tensor],
    params: torch.Tensor,
    bounds: Gaussian1DBounds,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
):
    """Evaluate 1D Gaussian for running speed."""
    rs, _ = xy_tuple
    return gaussian_1d(rs, params, bounds, min_sigma, optimiser=optimiser)


def gaussian_ratio(
    xy_tuple: tuple[torch.Tensor, torch.Tensor],
    params: torch.Tensor,
    bounds: Gaussian1DBounds,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
):
    """Evaluate 1D Gaussian for ratio of running speed to optic flow."""
    rs, of = xy_tuple
    # running speed and optic flow are already in log space
    ratio = rs - of
    return gaussian_1d(ratio, params, bounds, min_sigma, optimiser=optimiser)


def gaussian_additive(
    xy_tuple: tuple[torch.Tensor, torch.Tensor],
    params: torch.Tensor,
    bounds: GaussianAdditiveBounds,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
) -> torch.Tensor:
    """Evaluate additive 2D Gaussian (independent RS and OF tuning, summed, sharing a
    single offset) -- matches fit_gaussian_blob.gaussian_additive's math exactly.

    Args:
        xy_tuple: (rs, of) tensors, each of shape (n_samples,).
        params: Tensor of shape (n_rois, 7): [log_amplitude_x, log_amplitude_y, x0,
            y0, log_sigma_x2, log_sigma_y2, offset] -- order matches
            fit_gaussian_blob.GaussianAdditiveParams exactly.
        bounds: Bounds for the x0 (rs) and y0 (of) centers.
        min_sigma: Small additive term for variance stability.

    Returns:
        Tensor of shape (n_samples, n_rois).
    """
    rs, of = xy_tuple
    single = params.ndim == 1
    if single:
        params = params.unsqueeze(0)

    log_amplitude_x = params[:, 0]
    log_amplitude_y = params[:, 1]
    x0 = params[:, 2]
    y0 = params[:, 3]
    log_sigma_x2 = params[:, 4]
    log_sigma_y2 = params[:, 5]
    offset = params[:, 6]

    if optimiser is not None and optimiser.lower() == "adamw" and bounds is not None:
        x0 = torch_utils.bounded_sigmoid(x0, bounds.x0_min, bounds.x0_max)
        y0 = torch_utils.bounded_sigmoid(y0, bounds.y0_min, bounds.y0_max)

    sigma_x_sq = torch.exp(log_sigma_x2) + min_sigma
    sigma_y_sq = torch.exp(log_sigma_y2) + min_sigma
    amplitude_x = torch.exp(log_amplitude_x)
    amplitude_y = torch.exp(log_amplitude_y)

    delta_x = rs[:, None] - x0[None, :]
    delta_y = of[:, None] - y0[None, :]

    result = (
        offset[None, :]
        + amplitude_x[None, :] * torch.exp(-(delta_x**2) / (2.0 * sigma_x_sq[None, :]))
        + amplitude_y[None, :] * torch.exp(-(delta_y**2) / (2.0 * sigma_y_sq[None, :]))
    )
    return result.squeeze(-1) if single else result


def gaussian_multiplicative(
    xy_tuple: tuple[torch.Tensor, torch.Tensor],
    params: torch.Tensor,
    bounds: Gaussian2DBounds,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
):
    """Evaluate multiplicative 2D Gaussian for running speed and optic flow."""
    rs, of = xy_tuple
    x = rs - of  # fit the ratio of running speed to optic flow
    y = rs  # fit the running speed
    return gaussian_2mult(x, y, params, bounds, min_sigma, optimiser=optimiser)


def gaussian_2d_cholesky(
    xy_tuple: tuple[torch.Tensor, torch.Tensor],
    params: torch.Tensor,
    bounds: Gaussian2DCholeskyBounds,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
):
    """Evaluate 2D Gaussian for running speed and optic flow (Cholesky
    parameterisation).
    """
    rs, of = xy_tuple
    return gaussian_bivar_cholesky(
        rs, of, params, bounds, min_sigma, optimiser=optimiser
    )


def gaussian_2d(
    xy_tuple: tuple[torch.Tensor, torch.Tensor],
    params: torch.Tensor,
    bounds: Gaussian2DAngleBounds,
    min_sigma: float = 0.25,
    optimiser: str | None = None,
):
    """Evaluate 2D Gaussian for running speed and optic flow (angle/sigma
    parameterisation).
    """
    rs, of = xy_tuple
    return gaussian_bivar_angle(rs, of, params, bounds, min_sigma, optimiser=optimiser)


## Helper functions for munging the input data
MODEL_SPECS = {
    "gaussian_2d": gaussian_2d_cholesky,  # Cholesky parameterisation (active)
    # "gaussian_2d": gaussian_2d,  # angle/sigma parameterisation -- commented out,
    # uncomment (and comment out the line above) to rewire the angle
    # parameterisation back in
    "gaussian_RS": gaussian_rs,
    "gaussian_OF": gaussian_of,
    "gaussian_ratio": gaussian_ratio,
    "gaussian_multiplicative": gaussian_multiplicative,
    "gaussian_additive": gaussian_additive,
}

MODEL_ABBRV = {
    "gaussian_2d": "g2d",
    "gaussian_RS": "grs",
    "gaussian_OF": "gof",
    "gaussian_ratio": "gratio",
    "gaussian_multiplicative": "g2mult",
    "gaussian_additive": "gadd",
}

def process_rs_of_for_fit(
    trials_df: pd.DataFrame,
    trial_list: list = [],
    rs_col: str = "RS_volume_stim",
    of_col: str = "OF_stim",
    response_col: str = "dff_stim",
    rs_threshold: float = 0.01,
    max_acc: float | None = None,
    max_rs2motor_diff: float | None = None,
    min_valid_frames: int | None = None,
    trial_average: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten trial data into vectors for fitting.

    Args:
        trials_df: DataFrame with trial data.
        rs_col: Column name for RS values.
        of_col: Column name for OF values.
        response_col: Column name for response values.
        rs_threshold: Minimum RS value to include a sample.
        trial_average: If True, average responses across trials before returning.
    Returns:
        Tuple of (rs, of, rs_eye, responses, depth) where rs, of, and rs_eye are 1D arrays of stimuli,
        responses is a 2D array of shape (n_samples, n_rois), and depth is a 1D array of depths.
    """

    subset = trials_df.iloc[trial_list] if len(trial_list) > 0 else trials_df

    rs_list = []
    of_list = []
    rs_eye_list = []
    response_list = []
    depth_list = []

    for _, trial in subset.iterrows():
        rs = np.asarray(
            np.nanmean(trial[rs_col], axis=1) if "volume" in rs_col else trial[rs_col]
        )
        rs_eye = np.asarray(trial["RS_eye_stim"])
        of = np.asarray(trial[of_col])
        responses = np.asarray(trial[response_col])
        depths = np.full_like(rs, trial["depth"])

        # choose frames that are above the running speed threshold
        running = (rs > rs_threshold) & (~np.isnan(of)) & (of > 0)
        # remove frames/volumes that are outside the acceleration ratio
        if max_acc is not None and "acceleration_ratio_max_stim" in trials_df.columns:
            acc = trial["acceleration_ratio_max_stim"]
            running &= acc <= max_acc
        if (
            max_rs2motor_diff is not None
            and "max_abs_rs2motor_diff_ratio_stim" in trials_df.columns
        ):
            rs2motor_diff = trial["max_abs_rs2motor_diff_ratio_stim"]
            running &= rs2motor_diff <= max_rs2motor_diff
        if np.sum(running) == 0:
            warnings.warn(
                f"No valid frames for trial {trial.name} after applying thresholds."
            )
            continue
        if min_valid_frames is not None and np.sum(running) < min_valid_frames:
            continue
        if trial_average:
            rs_list.append(np.mean(rs[running]))
            rs_eye_list.append(np.mean(rs_eye[running]))
            of_list.append(np.mean(of[running]))
            response_list.append(np.mean(responses[running, :], axis=0))
            depth_list.append(np.mean(depths[running]))
        else:
            rs_list.append(rs[running])
            rs_eye_list.append(rs_eye[running])
            of_list.append(of[running])
            response_list.append(responses[running, :])
            depth_list.append(depths[running])

    rs = np.log(np.concatenate(rs_list))
    of = np.log(np.degrees(np.concatenate(of_list)))
    rs_eye = np.log(np.concatenate(rs_eye_list))

    return (
        rs,
        of,
        rs_eye,
        np.concatenate(response_list, axis=0),
        np.concatenate(depth_list),
    )


## Helper functions for computing fit metrics
def _r2_per_roi(
    y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Compute R^2 independently for each ROI column."""
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    y_mean = y_true.mean(dim=0, keepdim=True)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=0).clamp_min(eps)
    return 1.0 - ss_res / ss_tot


def _validate_and_filter_fit_arrays(
    rs: np.ndarray,
    of: np.ndarray,
    responses: np.ndarray,
    depth: np.ndarray,
    trial_average: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate stimulus/response array shapes and drop non-finite stimulus samples.

    Args:
        rs: 1D array of log-RS values.
        of: 1D array of log-OF values.
        responses: 2D array of shape (n_samples, n_rois).
        depth: 1D array of depths, one per sample.
        trial_average: If True, samples are already trial-averaged and are not
            filtered for finiteness.

    Returns:
        Tuple of (rs, of, responses, depth) with samples removed where rs or of
        is non-finite. Responses are not used to filter samples -- a frame is not
        dropped for every ROI just because one ROI has a non-finite response there
        (matches the notebook checkpoint's filtering, which only checks rs/of).
    """
    if rs.ndim != 1 or of.ndim != 1 or depth.ndim != 1:
        raise ValueError("RS and OF inputs must be 1D arrays.")
    if responses.ndim != 2:
        raise ValueError(
            "Responses input must be a 2D array with shape (n_samples, n_rois)."
        )
    if (
        rs.shape[0] != of.shape[0]
        or rs.shape[0] != responses.shape[0]
        or rs.shape[0] != depth.shape[0]
    ):
        raise ValueError(
            "RS, OF, depth, and responses must have the same number of samples."
        )
    if not trial_average:
        print("Validating stimulus arrays...", flush=True)
        valid = np.isfinite(rs) & np.isfinite(of)
        rs = rs[valid]
        of = of[valid]
        responses = responses[valid, :]
        depth = depth[valid]
    return rs, of, responses, depth


def _make_fit_tensors(
    rs: np.ndarray,
    of: np.ndarray,
    responses: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, int]:
    """Build the (X, y) tensors used by the torch fit classes.

    Returns:
        Tuple of (X, y, n_rois) where X is a tuple of (rs, of) tensors and y is
        the response tensor of shape (n_samples, n_rois).
    """
    X = (
        torch.tensor(rs, dtype=dtype, device=device),
        torch.tensor(of, dtype=dtype, device=device),
    )
    y = torch.tensor(responses, dtype=dtype, device=device)
    return X, y, y.shape[1]


def _fit_trf(
    X: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    y: torch.Tensor,
    model: str,
    model_func: Callable,
    bounds: Gaussian1DBounds | Gaussian2DAngleBounds | Gaussian2DCholeskyBounds | GaussianAdditiveBounds,
    param_range: dict[str, float],
    n_starts: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    curve_fit_config: CurveFitConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit all ROIs with n_starts random inits with TRF.
    
    Args:
        X: Stimulus tensor or tuple of tensors, shared across all ROIs/starts.
        y: Response tensor of shape (n_samples, n_rois).
        model: Model string (e.g. "gaussian_2d") -- see torch_utils.MODEL_N_PARAMS.
        model_func: Model function from MODEL_SPECS, already bound to a fixed
            `min_sigma` (e.g. via functools.partial).
        bounds: Bounds dataclass for `model`.
        n_starts: Number of AdamW random restarts per ROI.
        seed: RNG seed for initialisation.
        curve_fit_config: CurveFitConfig dataclass containing TRF refinement settings.

    Returns:
        Tuple of (best_params, best_r2): best_params has shape (n_rois, n_params) in
        natural/data space, best_r2 has shape (n_rois,).
    """
    n_rois = y.shape[1]
    # get the same inits as the scipy pipeline
    _, lower_bounds, upper_bounds, p0_func = initial_fit_conditions(
        model, param_range=param_range
    )

    X_np = tuple(x.detach().cpu().numpy() for x in X)
    y_np = y.detach().cpu().numpy()
    p0s = []
    for i in range(n_rois):
        tmp = []
        for j in range(n_starts):
            p0 = p0_func(X=X_np, y=y_np[:, i], i_iter=j)
            tmp.append(torch.tensor(p0, device=device, dtype=dtype))
        p0s.append(torch.stack(tmp))

    initial_params = torch.stack(p0s).reshape(n_rois * n_starts, -1)

    trf_fit = torch_utils.Curve_fit(
        X=X,
        y=y,
        params=initial_params,
        bounds=bounds,
        model_func=model_func,
        n_starts=n_starts,
        n_iters=curve_fit_config.n_iters,
        method=curve_fit_config.method,
    )
    params_fit = trf_fit.fit()
    r2_final = trf_fit.r2.view(n_rois, n_starts)
    best_r2, best_idx = r2_final.max(dim=1)
    best_params = params_fit.view(n_rois, n_starts, -1)[torch.arange(n_rois), best_idx]
    return best_params, best_r2

def _fit_adamw_then_curve(
    X: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    y: torch.Tensor,
    model: str,
    model_func: Callable,
    bounds: Gaussian2DBounds | Gaussian1DBounds | Gaussian2DCholeskyBounds,
    n_starts: int,
    top_k: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    adamw_config: AdamWFitConfig,
    curve_fit_config: CurveFitConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit all ROIs with AdamW random restarts, then refine the top-K starts per ROI with TRF.

    Args:
        X: Stimulus tensor or tuple of tensors, shared across all ROIs/starts.
        y: Response tensor of shape (n_samples, n_rois).
        model: Model string (e.g. "gaussian_2d") -- see torch_utils.MODEL_N_PARAMS.
        model_func: Model function from MODEL_SPECS, already bound to a fixed
            `min_sigma` (e.g. via functools.partial).
        bounds: Bounds dataclass for `model`.
        n_starts: Number of AdamW random restarts per ROI.
        top_k: Number of top (by AdamW R^2) restarts per ROI to refine with TRF.
        seed: RNG seed for initialisation.
        adamw_config: AdamWFitConfig dataclass containing AdamW random restart settings.
        curve_fit_config: CurveFitConfig dataclass containing TRF refinement settings.

    Returns:
        Tuple of (best_params, best_r2): best_params has shape (n_rois, n_params) in
        natural/data space, best_r2 has shape (n_rois,).
    """
    n_rois = y.shape[1]

    initial_params = torch_utils.generate_n_inits_all_rois(
        n_starts,
        X,
        y,
        model=model,
        bounds=bounds,
        rng_seed=seed,
        device=device,
        dtype=dtype,
        apply_bounds=True,
    )
    adamw_fit = torch_utils.AdamW_fit(
        X=X,
        y=y,
        params=torch.nn.Parameter(initial_params),
        bounds=bounds,
        model=model,
        model_func=model_func,
        n_steps=adamw_config.n_steps,
        n_starts=n_starts,
        lr=adamw_config.lr,
        weight_decay=adamw_config.weight_decay,
        loss_fn=adamw_config.loss_fn,
        smooth_l1_beta=adamw_config.smooth_l1_beta,
    )
    params_fit = adamw_fit.fit()
    # transform the parameters back into stimulus data space
    params_fit = torch_utils.decode_params(params_fit, model=model, bounds=bounds)
    params_fit = params_fit.view(n_rois, n_starts, -1)

    # get the top K parameters based on AdamW R^2
    _, topk_indices = torch.topk(adamw_fit.r2, top_k, dim=1)
    topk_popt = torch.gather(
        params_fit, 1, topk_indices.unsqueeze(-1).expand(-1, -1, params_fit.shape[-1])
    )
    topk_popt = topk_popt.view(n_rois * top_k, -1)

    # refine the top K parameters per ROI with TRF
    trf_fit = torch_utils.Curve_fit(
        X=X,
        y=y,
        params=topk_popt,
        bounds=bounds,
        model_func=model_func,
        n_starts=top_k,
        n_iters=curve_fit_config.n_iters,
        method=curve_fit_config.method,
    )
    params_fit = trf_fit.fit()
    r2_final = trf_fit.r2.view(n_rois, top_k)
    best_r2, best_idx = r2_final.max(dim=1)
    best_params = params_fit.view(n_rois, top_k, -1)[torch.arange(n_rois), best_idx]
    return best_params, best_r2


## Variable Projection (VarPro) analytic candidate initialisation for g2d
## (prototype -- see notebooks/pytorch_fits.ipynb for orchestration/validation
## against AdamW+TRF and scipy; not yet wired into fit_rs_of_tuning).
def generate_varpro_candidates(
    X: tuple[torch.Tensor, torch.Tensor],
    y: torch.Tensor,
    bounds: Gaussian2DCholeskyBounds,
    n_bins: int = 10,
    min_bin_count: int = 5,
    top_m_centers: int = 5,
    shape_grid: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Cheap, gradient-free candidate shape params for a single ROI's fit.

    Crosses the `top_m_centers` best-response (x0, y0) bin centers (via
    `torch_utils._binned_peak_guesses_topm`) with a small fixed grid of
    (log_l11, l21, log_l22) shape presets. Each candidate costs one forward
    pass to score (see `score_varpro_candidates`), not a gradient-descent
    trajectory, so `n_candidates` is intentionally much larger than AdamW's
    typical `n_starts` (e.g. 10).

    Args:
        X: (rs, of) tensors, each of shape (n_samples,).
        y: Response tensor of shape (n_samples,) for one ROI.
        bounds: Bounds for x0/y0, used only to clip degenerate centers.
        top_m_centers: Number of data-driven (x0, y0) peaks to seed.
        shape_grid: Optional (n_presets, 3) tensor of (log_l11, l21, log_l22)
            presets. Defaults to a small isotropic-scale x tilt grid.

    Returns:
        Tensor of shape (top_m_centers * n_presets, 5): natural-space
        [x0, y0, log_l11, l21, log_l22] candidates.
    """
    rs, of = X
    if device is None:
        device = rs.device
    if dtype is None:
        dtype = rs.dtype

    x0_centers, y0_centers = torch_utils._binned_peak_guesses_topm(
        rs, of, y, n_bins=n_bins, min_bin_count=min_bin_count, top_m=top_m_centers
    )
    x0_centers = x0_centers.clamp(bounds.x0_min, bounds.x0_max)
    y0_centers = y0_centers.clamp(bounds.y0_min, bounds.y0_max)

    if shape_grid is None:
        # A handful of isotropic scales (log_l11 == log_l22, l21 == 0 --
        # circular blobs at a few widths) crossed with a couple of tilts.
        log_scales = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
        tilts = torch.tensor([0.0, 0.5, -0.5], device=device, dtype=dtype)
        log_l11_grid, tilt_grid = torch.meshgrid(log_scales, tilts, indexing="ij")
        shape_grid = torch.stack(
            [log_l11_grid.reshape(-1), tilt_grid.reshape(-1), log_l11_grid.reshape(-1)],
            dim=1,
        )  # (n_presets, 3): [log_l11, l21, log_l22]
    shape_grid = shape_grid.to(device=device, dtype=dtype)

    n_centers = x0_centers.shape[0]
    n_presets = shape_grid.shape[0]
    x0_full = x0_centers.repeat_interleave(n_presets)
    y0_full = y0_centers.repeat_interleave(n_presets)
    shape_full = shape_grid.repeat(n_centers, 1)

    return torch.cat(
        [x0_full[:, None], y0_full[:, None], shape_full], dim=1
    )  # (n_centers * n_presets, 5)


def score_varpro_candidates(
    x: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    shape_candidates: torch.Tensor,
    min_sigma: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score candidate shapes by analytically solving for amplitude/offset
    (VarPro) and computing the resulting R^2 -- no gradient steps.

    Args:
        x, y: Stimulus tensors of shape (n_samples,) (rs, of).
        target: Response tensor of shape (n_samples,) for one ROI.
        shape_candidates: Tensor of shape (n_candidates, 5):
            [x0, y0, log_l11, l21, log_l22], as returned by
            `generate_varpro_candidates`.
        min_sigma: Small additive term for variance stability.

    Returns:
        Tuple of (offset, amplitude, r2), each of shape (n_candidates,).
        `amplitude` is clamped to a small positive floor before use elsewhere
        (e.g. `torch.log`) since VarPro's unconstrained linear solve can
        return non-positive values for a poorly-fitting shape -- such
        candidates simply score a low R^2 here and lose the subsequent
        top-k selection, no special-casing needed.
    """
    x0, y0, log_l11, l21, log_l22 = shape_candidates.unbind(dim=1)
    basis = _g2d_basis(x, y, x0, y0, log_l11, l21, log_l22, min_sigma=min_sigma)
    offset, amplitude = torch_utils.solve_linear_params(basis, target)
    amplitude = amplitude.clamp_min(1e-6)
    y_pred = offset[None, :] + amplitude[None, :] * basis
    r2 = torch_utils.calculate_r2(target[:, None].expand_as(y_pred), y_pred)
    return offset, amplitude, r2


def _per_roi_spearman(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Spearman rank correlation independently for each ROI/column.

    Args:
        y_true: Array of shape (n_samples, n_rois).
        y_pred: Array of shape (n_samples, n_rois).

    Returns:
        Tuple of (rval, pval), each a 1D array of length n_rois.
    """
    n_rois = y_true.shape[1]
    rval = np.empty(n_rois)
    pval = np.empty(n_rois)
    for roi in range(n_rois):
        rval[roi], pval[roi] = spearmanr(y_true[:, roi], y_pred[:, roi])
    return rval, pval


def _store_torch_fit_results(
    torch_df: pd.DataFrame,
    best_params: torch.Tensor,
    best_r2: torch.Tensor,
    rval: np.ndarray,
    pval: np.ndarray,
    protocol_sfx: str,
    rs_type: str,
    trial_sfx: str,
    model_sfx: str,
    min_sigma: float,
) -> None:
    """Assign single-fit params and fit-quality metrics into `torch_df` columns."""
    sfx = f"{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}_torch"
    torch_df[f"rsof_popt_{sfx}"] = best_params.detach().cpu().numpy().tolist()
    torch_df[f"rsof_rsq_{sfx}"] = best_r2.detach().cpu().numpy()
    torch_df[f"rsof_spearmanr_rval_{sfx}"] = rval
    torch_df[f"rsof_spearmanr_pval_{sfx}"] = pval
    torch_df[f"rsof_minSigma_{sfx}"] = min_sigma


def _store_torch_cv_results(
    torch_df: pd.DataFrame,
    train_r2_folds: list[np.ndarray],
    train_popt_folds: list[np.ndarray],
    train_rval_folds: list[np.ndarray],
    train_pval_folds: list[np.ndarray],
    test_r2: np.ndarray,
    test_rval: np.ndarray,
    test_pval: np.ndarray,
    protocol_sfx: str,
    rs_type: str,
    trial_sfx: str,
    model_sfx: str,
    min_sigma: float,
    k_folds: int,
    seed: int,
) -> None:
    """Assign per-fold train metrics and held-out test metrics into `torch_df` columns."""
    sfx = f"{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}_torch"
    train_r2_per_roi = np.stack(train_r2_folds, axis=1)  # (n_rois, k_folds)
    train_popt_per_roi = np.stack(
        train_popt_folds, axis=1
    )  # (n_rois, k_folds, n_params)
    train_rval_per_roi = np.stack(train_rval_folds, axis=1)  # (n_rois, k_folds)
    train_pval_per_roi = np.stack(train_pval_folds, axis=1)  # (n_rois, k_folds)

    torch_df[f"rsof_train_rsq_{sfx}"] = list(train_r2_per_roi)
    torch_df[f"rsof_train_popt_{sfx}"] = [p.tolist() for p in train_popt_per_roi]
    torch_df[f"rsof_train_spearmanr_rval_{sfx}"] = list(train_rval_per_roi)
    torch_df[f"rsof_train_spearmanr_pval_{sfx}"] = list(train_pval_per_roi)
    torch_df[f"rsof_test_rsq_{sfx}"] = test_r2
    torch_df[f"rsof_test_spearmanr_rval_{sfx}"] = test_rval
    torch_df[f"rsof_test_spearmanr_pval_{sfx}"] = test_pval
    torch_df[f"rsof_minSigma_{sfx}"] = min_sigma
    torch_df[f"rsof_randomState_{sfx}"] = seed
    torch_df[f"rsof_kFolds_{sfx}"] = k_folds


def fit_rs_of_tuning(
    trials_df,
    model: str = "gaussian_2d",
    use_col: str = "dff_stim",
    param_range: dict | None = None,
    choose_trials: list | None = None,
    trial_sfx: str = "",
    rs_thr: float = 0.01,
    n_starts: int = 5,
    # top_k: int = 3,
    k_folds: int = 1,
    seed: int = 42,
    run_closedloop_only: bool = False,
    run_openloop_only: bool = False,
    max_acc: Optional[float] = None,
    max_rs2motor_diff: Optional[float] = None,
    min_valid_frames: Optional[int] = None,
    trial_average: bool = False,
    min_sigma: float = 0.25,
    # adamw_lr: float = 0.05,
    # adamw_weight_decay: float = 1e-6,
    # adamw_n_steps: int = 1000,
    # adamw_loss_fn: str = "mse",
    # adamw_smooth_l1_beta: float = 1.0,
):
    """Run the RS/OF model fit on batches of neurons using PyTorch for gradient-based optimisation.

    Args:
        trials_df: DataFrame containing trial data with columns for running speed, optic flow, dF/F or spikes.
        model: Model type to fit. Options are 'gaussian_2d', 'gaussian_RS', 'gaussian_OF',
            'gaussian_ratio', 'gaussian_additive', 'gaussian_multiplicative'.
        use_col: Column name in trials_df to use as the target variable for fitting.
        choose_trials: Trials to include in the fit. Can be a list of trial indices. Defaults to None.
        trial_sfx: Suffix to append to saved column names in the output dataframe. Defaults to an empty string.
        rs_thr: Minimum running speed threshold for including data points in the fit. Defaults to 0.01.
        n_starts: Number of random AdamW initialisations per ROI. Defaults to 10
            (matches the notebook checkpoint's N_STARTS_ALL).
        top_k: Number of top (by AdamW R^2) starts per ROI to refine with TRF. Defaults to 3.
        k_folds: Number of folds for cross-validation. Defaults to 1.
        seed: Random seed for reproducibility. Defaults to 42.
        run_closedloop_only: If True, only include closed-loop trials in the fit. Defaults to False.
        run_openloop_only: If True, only include open-loop trials in the fit. Defaults to False.
        max_acc: Maximum acceleration threshold for including data points in the fit.
        max_rs2motor_diff: Maximum difference between running speed and motor speed for including data points in the fit.
        min_valid_frames: Minimum number of valid frames required for including a neuron in the fit. Defaults to None.
        trial_average: If True, average the data across trials before fitting. Defaults to False.
        min_sigma: Minimum variance floor passed to the model function. Defaults to 0.25.
        verbose: If True, print progress and fit metrics during the fitting process. Defaults to False.

    Returns:
        A DataFrame with the fitted parameters, R^2 values, and other fit metrics for each neuron.
    """

    # Set the boundary conditions for the chosen model
    if param_range is None:
        param_range = {
            "log_amplitude_max": 10.0,
            "rs_min": 0.005,
            "rs_max": 5,
            "of_min": 0.03,
            "of_max": 3000,
        }
    # adamw_config = AdamWFitConfig(
    #     lr=adamw_lr,
    #     weight_decay=adamw_weight_decay,
    #     n_steps=adamw_n_steps,
    #     loss_fn=adamw_loss_fn,
    #     smooth_l1_beta=adamw_smooth_l1_beta,
    # )
    refine_config = CurveFitConfig(
        n_iters=2000,
        method="trf",
    )
    bounds = torch_utils.format_model_bounds(
        model,
        **param_range,
    )

    # initialise torch device
    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device == "cpu":
        raise RuntimeError("CUDA is required for fitting.")
    # using double precision
    dtype = "float64"
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32

    # make a dataframe for saving the results
    torch_df = pd.DataFrame(
        columns=["roi"], data=np.arange(trials_df.iloc[0][use_col].shape[1], dtype=int)
    )
    model_abbrv = MODEL_ABBRV.get(model, model)
    model_sfx = "_" + model_abbrv

    # choose trials
    if choose_trials is not None and isinstance(choose_trials, list):
        (
            trials_df_select,
            choose_trial_nums,
            trial_sfx,
        ) = common_utils.choose_trials_subset(
            trials_df,
            choose_trials,
            sfx=trial_sfx,
        )
    else:
        trials_df_select = trials_df

    # loop through all protocols
    if run_closedloop_only:
        all_protocols = [1]
        print("Running closed-loop only fit...", flush=True)
    elif run_openloop_only:
        if 0 in trials_df_select.closed_loop.unique():
            all_protocols = [0]
            print("Running open-loop only fit...", flush=True)
        else:
            all_protocols = []
            print("Error: Open loop protocol not found!")
    else:
        all_protocols = [1] if (k_folds > 1) else trials_df_select.closed_loop.unique()
    assert len(all_protocols) <= 2, "More than two protocols detected!"

    for is_closedloop in all_protocols:
        protocol_sfx = "closedloop" if is_closedloop else "openloop"
        print(
            f"Process protocol {protocol_sfx}/{len(trials_df_select.closed_loop.unique())}..."
        )

        trials_df_fit = trials_df_select[
            trials_df_select.closed_loop == is_closedloop
        ].copy()
        if choose_trials is not None and isinstance(
            choose_trials, list
        ):  # a list of trials from all trials have already been chosen
            # choose only closed loop or open loop trials
            trials_df_fit = trials_df_fit
        else:  # Otherwise, if choose_trials is "even" or "odd", choose trials within a certain protocol
            (
                trials_df_fit,
                choose_trial_nums,
                trial_sfx,
            ) = common_utils.choose_trials_subset(
                trials_df_fit,
                choose_trials,
                sfx=trial_sfx,
            )

        trials_df_fit = depth_class_labels(trials_df_fit)
        depth_label = trials_df_fit["depth_label"].values

        # initialise a model function, bound to the requested variance floor
        model_func = functools.partial(MODEL_SPECS[model], min_sigma=min_sigma)

        rs_types_openloop = ["_actual", "_virtual"]
        if k_folds == 1:
            # process the data for fitting
            rs, of, rs_eye, responses, depth = process_rs_of_for_fit(
                trials_df_fit,
                trial_list=[],
                rs_col="RS_volume_stim",
                response_col=use_col,
                rs_threshold=rs_thr,
                max_acc=max_acc,
                max_rs2motor_diff=max_rs2motor_diff,
                trial_average=trial_average,
                min_valid_frames=min_valid_frames,
            )
            # loop between actual and virtual running speeds
            rs_arr = [rs]

            for i_rs, rs_to_use in enumerate(rs_arr):
                rs_type = "" if is_closedloop else rs_types_openloop[i_rs]
                print(f"Fitting {protocol_sfx}{rs_type} running...", flush=True)

                rs_valid, of_valid, responses_valid, depth_valid = (
                    _validate_and_filter_fit_arrays(
                        rs_to_use, of, responses, depth, trial_average
                    )
                )
                X, y, n_rois = _make_fit_tensors(
                    rs_valid, of_valid, responses_valid, torch_dtype, resolved_device
                )

                # best_params, best_r2 = _fit_adamw_then_refine(
                #     X,
                #     y,
                #     model,
                #     model_func,
                #     bounds,
                #     n_starts,
                #     top_k,
                #     seed,
                #     resolved_device,
                #     torch_dtype,
                #     adamw_config=adamw_config,
                #     refine_config=refine_config,
                # )

                best_params, best_r2 = _fit_trf(
                    X,
                    y,
                    model,
                    model_func,
                    bounds,
                    param_range,
                    n_starts,
                    seed,
                    resolved_device,
                    torch_dtype,
                    curve_fit_config=refine_config,
                )

                # calculate the spearman R and p-value per ROI
                y_pred = model_func(X, best_params, bounds, optimiser="trf")
                rval, pval = _per_roi_spearman(
                    y.cpu().numpy(), y_pred.detach().cpu().numpy()
                )
                _store_torch_fit_results(
                    torch_df,
                    best_params,
                    best_r2,
                    rval,
                    pval,
                    protocol_sfx,
                    rs_type,
                    trial_sfx,
                    model_sfx,
                    min_sigma,
                )

                # placeholder for a function that calculates the preferred running speed, optic flow and depth for each model

        if k_folds > 1:
            print(f"Fit with {k_folds} fold cross-validation...", flush=True)

            stratified_kfold = StratifiedKFold(
                n_splits=k_folds, shuffle=True, random_state=seed
            )

            rs_type = "" if is_closedloop else rs_types_openloop[0]
            print(
                f"Fitting {protocol_sfx}{rs_type} running with cross-validation...",
                flush=True,
            )

            train_r2_folds = []
            train_popt_folds = []
            train_rval_folds = []
            train_pval_folds = []
            test_pred_folds = []
            test_true_folds = []

            for fold, (train_idx, test_idx) in enumerate(
                stratified_kfold.split(np.zeros(len(depth_label)), depth_label)
            ):
                print(f"  Fold {fold + 1}/{k_folds}...", flush=True)
                rs_train, of_train, rs_eye_train, responses_train, depth_train = (
                    process_rs_of_for_fit(
                        trials_df_fit,
                        trial_list=train_idx,
                        rs_col="RS_volume_stim",
                        response_col=use_col,
                        rs_threshold=rs_thr,
                        max_acc=max_acc,
                        max_rs2motor_diff=max_rs2motor_diff,
                        trial_average=trial_average,
                    )
                )
                rs_test, of_test, rs_eye_test, responses_test, depth_test = (
                    process_rs_of_for_fit(
                        trials_df_fit,
                        trial_list=test_idx,
                        rs_col="RS_volume_stim",
                        response_col=use_col,
                        rs_threshold=rs_thr,
                        max_acc=max_acc,
                        max_rs2motor_diff=max_rs2motor_diff,
                        trial_average=trial_average,
                    )
                )

                rs_train, of_train, responses_train, depth_train = (
                    _validate_and_filter_fit_arrays(
                        rs_train, of_train, responses_train, depth_train, trial_average
                    )
                )
                rs_test, of_test, responses_test, depth_test = (
                    _validate_and_filter_fit_arrays(
                        rs_test, of_test, responses_test, depth_test, trial_average
                    )
                )

                X_train, y_train, n_rois = _make_fit_tensors(
                    rs_train, of_train, responses_train, torch_dtype, resolved_device
                )
                X_test, y_test, _ = _make_fit_tensors(
                    rs_test, of_test, responses_test, torch_dtype, resolved_device
                )

                # best_params, best_r2 = _fit_adamw_then_refine(
                #     X_train,
                #     y_train,
                #     model,
                #     model_func,
                #     bounds,
                #     n_starts,
                #     top_k,
                #     seed,
                #     resolved_device,
                #     torch_dtype,
                #     adamw_config=adamw_config,
                #     refine_config=refine_config,
                # )
                best_params, best_r2 = _fit_trf(
                    X_train,
                    y_train,
                    model,
                    model_func,
                    bounds,
                    param_range,
                    n_starts,
                    seed,
                    resolved_device,
                    torch_dtype,
                    curve_fit_config=refine_config,
                )
                y_test_pred = model_func(X_test, best_params, bounds, optimiser="trf")
                y_train_pred = model_func(X_train, best_params, bounds, optimiser="trf")
                train_rval, train_pval = _per_roi_spearman(
                    y_train.cpu().numpy(), y_train_pred.detach().cpu().numpy()
                )

                train_r2_folds.append(best_r2.detach().cpu().numpy())
                train_popt_folds.append(best_params.detach().cpu().numpy())
                train_rval_folds.append(train_rval)
                train_pval_folds.append(train_pval)
                test_pred_folds.append(y_test_pred.detach().cpu().numpy())
                test_true_folds.append(y_test.cpu().numpy())

            # pool held-out predictions across folds before scoring, so every sample
            # contributes to one test metric per ROI (matches fit_gaussian_blob.py)
            test_true_all = np.concatenate(test_true_folds, axis=0)
            test_pred_all = np.concatenate(test_pred_folds, axis=0)
            test_r2 = _r2_per_roi(
                torch.as_tensor(test_true_all), torch.as_tensor(test_pred_all)
            ).numpy()
            test_rval, test_pval = _per_roi_spearman(test_true_all, test_pred_all)

            _store_torch_cv_results(
                torch_df,
                train_r2_folds,
                train_popt_folds,
                train_rval_folds,
                train_pval_folds,
                test_r2,
                test_rval,
                test_pval,
                protocol_sfx,
                rs_type,
                trial_sfx,
                model_sfx,
                min_sigma,
                k_folds,
                seed,
            )

    return torch_df
