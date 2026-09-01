## Class for model specification

from dataclasses import dataclass
from typing import Callable
from tqdm import tqdm

import numpy as np
import torch
import functools

import warnings


## Classes for setting up boundary conditions
@dataclass
class Gaussian1DBounds:
    """Bounds for interpretable Gaussian parameters in log-space units.

    The fitted parameter vector order is:
    [log_amplitude, x0, log_sigma_x2, offset]

    Use these parameters for fitting grs and gof models.
    """

    log_amplitude_max: float
    x0_min: float
    x0_max: float


@dataclass
class Gaussian2DBounds:
    """Bounds for interpretable Gaussian parameters in log-space units.

    The fitted parameter vector order is:
    [log_amplitude, x0, y0, log_sigma_x2, log_sigma_y2, theta, offset]

    Use all parameters for fitting g2d and g2mult models. Use just x0 and y0 related
    params for gadd.
    """

    log_amplitude_max: float
    x0_min: float
    x0_max: float
    y0_min: float
    y0_max: float
    theta_min: float | None = 0.0
    theta_max: float | None = np.pi / 2


@dataclass
class Gaussian2DCholeskyBounds:
    """Bounds for Gaussian 2D models with Cholesky parameterisation."""

    x0_min: float
    x0_max: float
    y0_min: float
    y0_max: float
    log_amplitude_max: float
    l21_max: float = 1e4
    log_l_max: float = 15.0


@dataclass
class Gaussian2DAngleBounds:
    """Bounds for the angle/sigma-parameterised 2D Gaussian, matching the notebook
    checkpoint's `trf_batch_fit` exactly.

    The fitted parameter vector order is:
    [log_amplitude, x0, y0, log_sigma_x2, log_sigma_y2, theta, offset]

    log_amplitude, log_sigma_x2, log_sigma_y2, and offset are left unbounded (as in
    the checkpoint); only x0, y0, and theta are box-constrained.
    """

    x0_min: float
    x0_max: float
    y0_min: float
    y0_max: float
    log_amplitude_max: float
    theta_min: float = 0.0
    theta_max: float = np.pi / 2


@dataclass
class GaussianAdditiveBounds:
    """Bounds for the additive (independent RS + OF tuning, summed) Gaussian model.

    The fitted parameter vector order is:
    [log_amplitude_x, x0, y0, log_amplitude_y, log_sigma_x2, log_sigma_y2, offset]

    Matches fit_gaussian_blob.GaussianAdditiveParams's bounds exactly: only x0
    (rs) and y0 (of) are box-constrained; both amplitudes, both log-sigmas, and
    the shared offset are left unbounded.
    """

    x0_min: float
    x0_max: float
    y0_min: float
    y0_max: float


## Helper functions for formatting bounds as lower, upper for each type of model
def format_model_bounds(
    model: str,
    of_min: float,
    of_max: float,
    rs_min: float,
    rs_max: float,
    log_amplitude_max: float,
    theta_min: float | None = None,
    theta_max: float | None = None,
    l21_max: float | None = None,
    log_l_max: float | None = None,
) -> (
    Gaussian2DBounds
    | Gaussian1DBounds
    | Gaussian2DCholeskyBounds
    | Gaussian2DAngleBounds
    | GaussianAdditiveBounds
):
    """Format bounds for the specified model type."""
    if model == "g2d":
        # --- Cholesky g2d (active) ---
        return Gaussian2DCholeskyBounds(
            x0_min=np.log(rs_min),
            x0_max=np.log(rs_max),
            y0_min=np.log(of_min),
            y0_max=np.log(of_max),
            log_amplitude_max=log_amplitude_max,
            l21_max=l21_max if l21_max is not None else 1e4,
            log_l_max=log_l_max if log_l_max is not None else 15.0,
        )
        # --- angle/sigma-parameterised g2d (commented out; uncomment to rewire
        # the angle parameterisation back in, and comment out the Cholesky
        # branch above) -- matches the notebook checkpoint's trf_batch_fit
        # bounds exactly ---
        # return Gaussian2DAngleBounds(
        #     x0_min=np.log(rs_min),
        #     x0_max=np.log(rs_max),
        #     y0_min=np.log(of_min),
        #     y0_max=np.log(of_max),
        #     log_amplitude_max=log_amplitude_max,
        #     theta_min=theta_min if theta_min is not None else 0.0,
        #     theta_max=theta_max if theta_max is not None else np.pi / 2,
        # )
    elif model == "g2mult":
        # convert x params to minimum and maximum virtual depth
        return Gaussian2DBounds(
            log_amplitude_max=log_amplitude_max,
            x0_min=np.log(rs_min / of_max),
            x0_max=np.log(rs_max / of_min),
            y0_min=rs_min,
            y0_max=rs_max,
            theta_min=None,
            theta_max=None,
        )
    elif model == "gadd":
        return GaussianAdditiveBounds(
            x0_min=np.log(rs_min),
            x0_max=np.log(rs_max),
            y0_min=np.log(of_min),
            y0_max=np.log(of_max),
        )
    elif model == "grs":
        return Gaussian1DBounds(
            log_amplitude_max=log_amplitude_max,
            x0_min=np.log(rs_min),
            x0_max=np.log(rs_max),
        )
    elif model == "gof":
        return Gaussian1DBounds(
            log_amplitude_max=log_amplitude_max,
            x0_min=np.log(of_min),
            x0_max=np.log(of_max),
        )
    elif model == "gratio":
        return Gaussian1DBounds(
            x0_min=np.log(rs_min / of_max),
            x0_max=np.log(rs_max / of_min),
            log_amplitude_max=log_amplitude_max,
        )
    else:
        raise ValueError(f"Unknown model type: {model}")


def vectorise_bounds(
    bounds: (
        Gaussian2DBounds
        | Gaussian1DBounds
        | Gaussian2DCholeskyBounds
        | Gaussian2DAngleBounds
        | GaussianAdditiveBounds
    ),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Format bounds as lower and upper tensors for each parameter."""
    if isinstance(bounds, Gaussian2DBounds):
        lower = torch.tensor(
            [
                float("-inf"),  # log_amplitude
                bounds.x0_min,
                bounds.y0_min,
                float("-inf"),  # log_sigma_x2
                float("-inf"),  # log_sigma_y2
                float("-inf"),  # offset
            ]
        )
        upper = torch.tensor(
            [
                bounds.log_amplitude_max,  # log_amplitude
                bounds.x0_max,
                bounds.y0_max,
                float("inf"),  # log_sigma_x2
                float("inf"),  # log_sigma_y2
                float("inf"),  # offset
            ]
        )
    elif isinstance(bounds, Gaussian1DBounds):
        lower = torch.tensor(
            [
                float("-inf"),  # log_amplitude
                bounds.x0_min,
                float("-inf"),  # log_sigma_x2
                float("-inf"),  # offset
            ]
        )
        upper = torch.tensor(
            [
                bounds.log_amplitude_max,  # log_amplitude
                bounds.x0_max,
                float("inf"),  # log_sigma_x2
                float("inf"),  # offset
            ]
        )
    elif isinstance(bounds, Gaussian2DCholeskyBounds):
        lower = torch.tensor(
            [
                float("-inf"),  # log_amplitude
                bounds.x0_min,
                bounds.y0_min,
                -bounds.log_l_max,  # log_l11
                -bounds.l21_max,  # l21
                -bounds.log_l_max,  # log_l22
                float("-inf"),  # offset
            ]
        )
        upper = torch.tensor(
            [
                bounds.log_amplitude_max,  # log_amplitude
                bounds.x0_max,
                bounds.y0_max,
                bounds.log_l_max,  # log_l11
                bounds.l21_max,  # l21
                bounds.log_l_max,  # log_l22
                float("inf"),  # offset
            ]
        )
    elif isinstance(bounds, Gaussian2DAngleBounds):
        lower = torch.tensor(
            [
                float("-inf"),  # log_amplitude
                bounds.x0_min,
                bounds.y0_min,
                float("-inf"),  # log_sigma_x2
                float("-inf"),  # log_sigma_y2
                bounds.theta_min,
                float("-inf"),  # offset
            ]
        )
        upper = torch.tensor(
            [
                float("inf"),
                bounds.x0_max,
                bounds.y0_max,
                float("inf"),  # log_sigma_x2
                float("inf"),  # log_sigma_y2
                bounds.theta_max,
                float("inf"),  # offset
            ]
        )
    elif isinstance(bounds, GaussianAdditiveBounds):
        # order matches fit_gaussian_blob.GaussianAdditiveParams exactly:
        # [log_amplitude_x, log_amplitude_y, x0, y0, log_sigma_x2, log_sigma_y2, offset]
        lower = torch.tensor(
            [
                float("-inf"),  # log_amplitude_x
                float("-inf"),  # log_amplitude_y
                bounds.x0_min,
                bounds.y0_min,
                float("-inf"),  # log_sigma_x2
                float("-inf"),  # log_sigma_y2
                float("-inf"),  # offset
            ]
        )
        upper = torch.tensor(
            [
                float("inf"),  # log_amplitude_x
                float("inf"),  # log_amplitude_y
                bounds.x0_max,
                bounds.y0_max,
                float("inf"),  # log_sigma_x2
                float("inf"),  # log_sigma_y2
                float("inf"),  # offset
            ]
        )
    else:
        raise ValueError(f"Unknown bounds type: {type(bounds)}")
    return lower, upper


## Helper functions for mapping between unconstrained and bounded parameters
def invert_bounded_sigmoid(
    value: torch.Tensor, low: float, high: float, eps: float = 1e-4
) -> torch.Tensor:
    """Invert the mapping from bounded_sigmoid."""
    p = (value - low) / (high - low)
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p / (1 - p))


def bounded_sigmoid(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Map unconstrained values to [low, high] with sigmoid."""
    return low + (high - low) * torch.sigmoid(raw)


def bounded_softplus_upper(raw: torch.Tensor, high: float) -> torch.Tensor:
    """Map unconstrained values to (-inf, high] with a smooth one-sided cap.
    """
    return high - torch.nn.functional.softplus(high - raw)


def invert_bounded_softplus_upper(
    value: torch.Tensor, high: float, eps: float = 1e-4
) -> torch.Tensor:
    """Invert the mapping from bounded_softplus_upper."""
    gap = torch.clamp(high - value, min=eps)
    return high - torch.log(torch.expm1(gap))


MODEL_N_PARAMS = {
    "g2d": 7,  # log_amplitude, x0, y0, log_l11, l21, log_l22, offset (Cholesky
    # parameterisation, active; angle/sigma layout -- log_sigma_x2, log_sigma_y2,
    # theta instead of log_l11, l21, log_l22 -- commented out alongside its
    # callsites in format_model_bounds/decode_params/_default_init)
    "g2mult": 6,  # log_amplitude, x0, y0, log_sigma_x2, log_sigma_y2, offset
    "gadd": 7,  # log_amplitude_x, log_amplitude_y, x0, y0, log_sigma_x2, log_sigma_y2,
    # offset -- order matches fit_gaussian_blob.GaussianAdditiveParams exactly
    "grs": 4,  # log_amplitude, x0, log_sigma_x2, offset
    "gof": 4,
    "gratio": 4,
}


def decode_params(
    raw_params: torch.Tensor,
    model: str,
    bounds: Gaussian2DBounds | Gaussian1DBounds | Gaussian2DCholeskyBounds,
) -> torch.Tensor:
    """Decode unconstrained (sigmoid pre-image) raw parameters into data space.
    (Inverse of the mapping used in `generate_n_inits` for initialisation.)
    """
    natural = raw_params.clone()
    if model == "gadd":
        # x0/y0 sit at indices 2/3 here (log_amplitude_x, log_amplitude_y come first),
        # matching fit_gaussian_blob.GaussianAdditiveParams's order exactly.
        natural[:, 2] = bounded_sigmoid(raw_params[:, 2], bounds.x0_min, bounds.x0_max)
        natural[:, 3] = bounded_sigmoid(raw_params[:, 3], bounds.y0_min, bounds.y0_max)
        return natural
    natural[:, 1] = bounded_sigmoid(raw_params[:, 1], bounds.x0_min, bounds.x0_max)
    if model in ("g2d", "g2mult"):
        natural[:, 2] = bounded_sigmoid(raw_params[:, 2], bounds.y0_min, bounds.y0_max)
    if model == "g2d":
        # --- Cholesky g2d (active) ---
        natural[:, 0] = bounded_softplus_upper(raw_params[:, 0], bounds.log_amplitude_max)
        natural[:, 3] = bounded_sigmoid(
            raw_params[:, 3], -bounds.log_l_max, bounds.log_l_max
        )
        natural[:, 4] = bounded_sigmoid(
            raw_params[:, 4], -bounds.l21_max, bounds.l21_max
        )
        natural[:, 5] = bounded_sigmoid(
            raw_params[:, 5], -bounds.log_l_max, bounds.log_l_max
        )
        # --- angle/sigma-parameterised g2d 
        # natural[:, 5] = bounded_sigmoid(raw_params[:, 5], bounds.theta_min, bounds.theta_max)
    return natural


## Helper functions for assigning random or data-informed initial values for fitting
def _default_init(
    n_rois: int,
    model: str,
    rng_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create raw parameter initialisation for batched optimisation."""

    g = torch.Generator(device=device)
    g.manual_seed(rng_seed)

    n_params = MODEL_N_PARAMS[model]
    raw = torch.zeros((n_rois, n_params), device=device, dtype=dtype)

    if model == "gadd":
        # [log_amplitude_x, log_amplitude_y, x0, y0, log_sigma_x2, log_sigma_y2, offset]
        # -- diverges from the shared layout below (x0 isn't at index 1), so it gets
        # its own self-contained init rather than slotting into the is_2d branch.
        for i in range(n_params):
            raw[:, i] = (
                torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.1
            )
        return raw

    is_2d = n_params >= 6  # "g2d" (7) and "g2mult" (6) both have an (x0, y0) centre

    raw[:, 0] = (
        torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.1
    )  # log_amplitude
    raw[:, 1] = torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.1  # x0

    if is_2d:
        raw[:, 2] = (
            torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.1
        )  # y0
        if model == "g2d":
            # --- Cholesky g2d init (active) ---
            raw[:, 3] = (
                -2.0 + torch.rand(n_rois, device=device, dtype=dtype, generator=g) * 4.0
            )  # log_l11
            raw[:, 4] = (
                torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.5
            )  # l21
            raw[:, 5] = (
                -2.0 + torch.rand(n_rois, device=device, dtype=dtype, generator=g) * 4.0
            )  # log_l22
            # --- angle/sigma-parameterised g2d init (commented out; uncomment to
            # rewire the angle parameterisation back in, and comment out the
            # Cholesky block above) -- matches the notebook checkpoint's
            # _default_init exactly (N(0, 0.2), not 0.5) ---
            # raw[:, 3] = (
            #     torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.2
            # )  # log_sigma_x2
            # raw[:, 4] = (
            #     torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.2
            # )  # log_sigma_y2
            # raw[:, 5] = (
            #     torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.2
            # )  # theta (pre-sigmoid) -- never overwritten downstream, unlike x0/y0
        else:  # g2mult
            raw[:, 3] = (
                torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.1
            )  # log_sigma_x2
            raw[:, 4] = (
                torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.1
            )  # log_sigma_y2
        raw[:, -1] = (
            torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.1
        )  # offset
    else:
        raw[:, 2] = (
            torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.1
        )  # log_sigma_x2
        raw[:, 3] = (
            torch.randn(n_rois, device=device, dtype=dtype, generator=g) * 0.1
        )  # offset

    return raw


def _make_data_driven_guess(
    x: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 10,
    min_bin_count: int = 5,
) -> torch.Tensor:
    """Make a data-driven guess for the initial parameter based on the target response."""

    finite = torch.isfinite(x)
    x_valid = x[finite]
    t_valid = target[finite]

    x_edges = torch.linspace(
        x_valid.min(), x_valid.max(), n_bins + 1, device=x.device, dtype=x.dtype
    )
    x_bin = torch.clamp(torch.bucketize(x_valid, x_edges) - 1, 0, n_bins - 1)
    bin_sum = torch.zeros(n_bins, device=x.device, dtype=x.dtype).scatter_add_(
        0, x_bin, t_valid
    )
    bin_count = torch.zeros(n_bins, device=x.device, dtype=x.dtype).scatter_add_(
        0, x_bin, torch.ones_like(t_valid)
    )

    is_enough = bin_count >= min_bin_count
    if not torch.any(is_enough):
        # If no bins have enough data, return the mean of all valid target values
        return t_valid.mean()
    bin_mean = torch.where(
        is_enough,
        bin_sum / bin_count.clamp_min(1),
        torch.full_like(bin_sum, float("-inf")),
    )
    best_bin = torch.argmax(bin_mean)
    return x_valid[best_bin].mean()


def _binned_peak_guess(
    x: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 10,
    min_bin_count: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Data-informed (x0, y0) guess: bin samples by (x, y), average the response in
    each bin, and center on the bin with the highest mean -- more robust than
    anchoring on a single (possibly noisy) peak sample.
    """
    finite = torch.isfinite(target)
    x_valid = x[finite]
    y_valid = y[finite]
    t_valid = target[finite]

    x_edges = torch.linspace(
        x_valid.min(), x_valid.max(), n_bins + 1, device=x.device, dtype=x.dtype
    )
    y_edges = torch.linspace(
        y_valid.min(), y_valid.max(), n_bins + 1, device=y.device, dtype=y.dtype
    )

    x_bin = torch.clamp(torch.bucketize(x_valid, x_edges[1:-1]), 0, n_bins - 1)
    y_bin = torch.clamp(torch.bucketize(y_valid, y_edges[1:-1]), 0, n_bins - 1)
    bin_id = x_bin * n_bins + y_bin

    bin_sum = torch.zeros(n_bins * n_bins, dtype=t_valid.dtype, device=t_valid.device)
    bin_count = torch.zeros(n_bins * n_bins, dtype=t_valid.dtype, device=t_valid.device)
    bin_sum.scatter_add_(0, bin_id, t_valid)
    bin_count.scatter_add_(0, bin_id, torch.ones_like(t_valid))

    # require a minimum count so a single outlier sample can't define its own bin;
    # fall back to any non-empty bin if none meet the threshold (e.g. sparse data).
    enough = bin_count >= min_bin_count
    if not torch.any(enough):
        enough = bin_count >= 1
    bin_mean = torch.where(
        enough,
        bin_sum / bin_count.clamp_min(1),
        torch.full_like(bin_sum, float("-inf")),
    )
    best_bin = torch.argmax(bin_mean)

    in_best_bin = bin_id == best_bin
    return x_valid[in_best_bin].mean(), y_valid[in_best_bin].mean()


def generate_n_inits(
    n_starts: int,
    X: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    y: torch.Tensor | None,
    model: str,
    bounds: Gaussian2DBounds | Gaussian1DBounds | Gaussian2DCholeskyBounds,
    rng_seed: int,
    device: torch.device,
    dtype: torch.dtype,
    apply_bounds: bool = False,
) -> torch.Tensor:
    """Generate n_starts initial parameter sets for fitting.

    Args:
        X: Tuple of (rs, of) stimulus tensors, each of shape (n_samples,) -- same
            convention as `gaussian_rs`/`gaussian_of`/`gaussian_ratio`/`gaussian_2d`.
            For 1D models the relevant projection (rs, of, or rs - of) is extracted
            internally.
        y: Response values for the ROI being fit, shape (n_samples,), or None. If
            given, start 0 is anchored at the (x, y) center of the best-response bin
            (see `_binned_peak_guess` for 2D models, `_make_data_driven_guess` for
            1D models) instead of an uninformed random guess.
        apply_bounds: 
            - True: run through `invert_bounded_sigmoid`, allowing the fit parameter to be 
              unbounded.
            - False (default): written directly, unmodified, in data space units 
    """
    is_2d = MODEL_N_PARAMS[model] >= 6
    if is_2d:
        x_stim, y_stim = X
    elif model == "grs":
        x_stim, y_stim = X[0], None
    elif model == "gof":
        x_stim, y_stim = X[1], None
    elif model == "gratio":
        x_stim, y_stim = X[0] - X[1], None
    else:
        raise ValueError(f"Unknown model type: {model}")

    raw = _default_init(
        n_starts,
        model=model,
        rng_seed=rng_seed,
        device=device,
        dtype=dtype,
    )

    g = torch.Generator(device=device)
    g.manual_seed(rng_seed)

    # Choose x0 (and y0, for 2D bounds) values within the parameter range for x0 or y0
    x_range = bounds.x0_max - bounds.x0_min
    x0_targets = torch.zeros(n_starts, device=device, dtype=dtype)
    x0_targets[1:] = (
        bounds.x0_min
        + torch.rand(n_starts - 1, device=device, dtype=dtype, generator=g) * x_range
    )
    if is_2d:
        y_range = bounds.y0_max - bounds.y0_min
        y0_targets = torch.zeros(n_starts, device=device, dtype=dtype)
        y0_targets[1:] = (
            bounds.y0_min
            + torch.rand(n_starts - 1, device=device, dtype=dtype, generator=g)
            * y_range
        )

    if y is not None:
        # Anchor start 0 at the best-response location instead of a random guess.
        if is_2d:
            x0_targets[0], y0_targets[0] = _binned_peak_guess(x_stim, y_stim, y)
        else:
            x0_targets[0] = _make_data_driven_guess(x_stim, y)

    if apply_bounds:
        if model == "gadd":
            # x0/y0 sit at indices 2/3 here -- see decode_params/_default_init.
            raw[:, 2] = invert_bounded_sigmoid(x0_targets, bounds.x0_min, bounds.x0_max)
            raw[:, 3] = invert_bounded_sigmoid(y0_targets, bounds.y0_min, bounds.y0_max)
        else:
            raw[:, 1] = invert_bounded_sigmoid(x0_targets, bounds.x0_min, bounds.x0_max)
            if is_2d:
                raw[:, 2] = invert_bounded_sigmoid(
                    y0_targets, bounds.y0_min, bounds.y0_max
                )
    else:
        # no sigmoid transform
        if model == "gadd":
            raw[:, 2] = x0_targets
            raw[:, 3] = y0_targets
        else:
            raw[:, 1] = x0_targets
            if is_2d:
                raw[:, 2] = y0_targets

    return raw


def generate_n_inits_all_rois(
    n_starts: int,
    X: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    y: torch.Tensor,
    model: str,
    bounds: Gaussian2DBounds | Gaussian1DBounds | Gaussian2DCholeskyBounds,
    rng_seed: int,
    device: torch.device,
    dtype: torch.dtype,
    apply_bounds: bool = False,
) -> torch.Tensor:
    """Build (n_rois * n_starts, n_params) tensor of initialisations for fitting.

    Args:
        X: Stimulus values, as in `generate_n_inits`.
        y: Response matrix of shape (n_samples, n_rois) -- one column per ROI.
    """
    n_rois = y.shape[1]
    per_roi_inits = [
        generate_n_inits(
            n_starts,
            X,
            y[:, roi],
            model=model,
            bounds=bounds,
            rng_seed=rng_seed + roi,
            device=device,
            dtype=dtype,
            apply_bounds=apply_bounds,
        )
        for roi in range(n_rois)
    ]
    return torch.cat(per_roi_inits, dim=0)


## Helper function for Cholesky parameterisation of 2D Gaussian
def regularised_precision(
    log_l11: torch.Tensor,
    l21: torch.Tensor,
    log_l22: torch.Tensor,
    min_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the regularised precision matrix from Cholesky parameters."""
    log_l11 = log_l11.clamp(-170.0, 170.0)
    log_l22 = log_l22.clamp(-170.0, 170.0)
    l21 = l21.clamp(-1e150, 1e150)
    l11 = torch.exp(log_l11)
    l22 = torch.exp(log_l22)
    a0 = l11**2
    b0 = l11 * l21
    c0 = l21**2 + l22**2
    # Add min_sigma
    if min_sigma > 0:
        det0 = (l11 * l22) ** 2
        s = min_sigma
        D = 1 + 2 * s * (a0 + c0) + 4 * s**2 * det0
        a = (a0 + 2 * s * det0) / D
        b = b0 / D
        c = (c0 + 2 * s * det0) / D
    else:
        a, b, c = a0, b0, c0
    return a, b, c


## Helper function for calculating residuals with penalty, R-squared of the fit
def calculate_residual(
    params: torch.Tensor,
    X: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    y: torch.Tensor,
    model_func: Callable,
    bounds: Gaussian2DBounds | Gaussian1DBounds | Gaussian2DCholeskyBounds,
    optimiser: str | None,
    penalty: float,
) -> torch.Tensor:
    """Calculate the residuals between the model prediction and the target with penalty."""
    prediction = model_func(X, params, bounds, optimiser=optimiser)
    prediction = torch.nan_to_num(
        prediction, nan=penalty, posinf=penalty, neginf=-penalty
    )

    return prediction - y


def calculate_r2(
    y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Calculate the R-squared value for the fit."""
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    r2 = 1 - (ss_res / (ss_tot + eps))
    return r2


## Classes for AdamW optimiser and TRF optimiser
class AdamW_fit:
    """Class for AdamW fit object."""

    def __init__(
        self,
        X: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        y: torch.Tensor,
        params: torch.Tensor,
        model: str,
        bounds: Gaussian2DBounds | Gaussian1DBounds | Gaussian2DCholeskyBounds | Gaussian2DAngleBounds | GaussianAdditiveBounds,
        model_func: Callable,
        n_steps: int = 1000,
        n_starts: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        log_every: int = 1000,
        loss_fn: str = "mse",
        smooth_l1_beta: float = 1.0,
    ):
        self.X = X  # tuple of (x, y) tensors or single tensor depending on model
        self.y = y  # tensor of responses
        self.n_steps = n_steps
        self.params = params
        self.model = model
        self.bounds = bounds
        self.model_func = model_func
        self.n_starts = n_starts
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = params.device
        self.dtype = params.dtype
        self.log_every = log_every
        if loss_fn not in ("mse", "smooth_l1"):
            raise ValueError(f"Unknown loss_fn: {loss_fn!r} (expected 'mse' or 'smooth_l1')")
        self.loss_fn = loss_fn
        self.smooth_l1_beta = smooth_l1_beta
        self.initialise_optimiser()

    def _per_col_loss(self, z_pred: torch.Tensor, y_expanded: torch.Tensor) -> torch.Tensor:
        """Per-fit (per-column) training loss, averaged over samples.

        Note this is the loss AdamW optimises, distinct from the R^2/SSE metrics used
        for evaluation and for the Refine (LM/TRF) stage, which always use plain
        least-squares regardless of this setting -- switching AdamW to smooth_l1
        changes what basin it settles into, not what "good fit" means downstream.
        """
        if self.loss_fn == "mse":
            return torch.mean((z_pred - y_expanded) ** 2, dim=0)
        # smooth_l1: quadratic for |residual| < beta, linear beyond -- less sensitive
        # to outlier residuals than MSE. beta is in the same units as the response
        # (dff), so pick it relative to the residual scale you actually see; at
        # beta=1.0 (the torch default) it may behave almost identically to MSE if
        # typical residuals are well under 1.
        elementwise = torch.nn.functional.smooth_l1_loss(
            z_pred, y_expanded, reduction="none", beta=self.smooth_l1_beta
        )
        return torch.mean(elementwise, dim=0)

    def _initialise_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=self.n_steps, eta_min=1e-5
        )

    def initialise_optimiser(self):
        self.params = torch.nn.Parameter(self.params)
        self.optimiser = torch.optim.AdamW(
            [self.params],
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-6,
        )
        self._initialise_scheduler()

    def fit(self):
        params_all = self.params
        loss_history_all = torch.empty(
            (self.n_steps, params_all.shape[0]), device=self.device, dtype=self.dtype
        )
        # expand the y tensor to match the number of starts for each ROI
        y_expanded = self.y.repeat_interleave(self.n_starts, dim=1)
        # do the fit
        for step in tqdm(
            range(self.n_steps), desc="[batch-adamw] fitting all ROIs", unit="step"
        ):
            self.optimiser.zero_grad()
            z_pred = self.model_func(
                self.X, self.params, self.bounds, optimiser="adamw"
            )
            per_col_loss = self._per_col_loss(z_pred, y_expanded)
            loss = per_col_loss.mean()
            loss.backward()
            self.optimiser.step()
            self.scheduler.step()
            loss_history_all[step] = per_col_loss.detach()
            if step % self.log_every == 0 or step == self.n_steps - 1:
                current_lr = self.scheduler.get_last_lr()[0]
                print(
                    f"[batch-adamw] step {step}: mean loss = {loss.item():.6f}, lr = {current_lr:.5f}"
                )

        torch.cuda.synchronize()

        # Calculate the R-squared value for each fit
        with torch.no_grad():
            z_pred_final = self.model_func(
                self.X, self.params, self.bounds, optimiser="adamw"
            )
            r2_all = calculate_r2(y_expanded, z_pred_final)
            # add R-squared and loss history to the object for later inspection
            self.r2 = r2_all.detach().view(self.y.shape[1], self.n_starts)
            self.loss_history = loss_history_all.detach().view(
                self.n_steps, self.y.shape[1], self.n_starts
            )

        return self.params.detach()


class Refine_fit:
    """Class for TRF or LM fit object."""

    def __init__(
        self,
        X: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        y: torch.Tensor,
        params: torch.Tensor,
        bounds: Gaussian2DBounds | Gaussian1DBounds | Gaussian2DCholeskyBounds | Gaussian2DAngleBounds | GaussianAdditiveBounds,
        model_func: Callable,
        n_starts: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        n_iters: int = 1000,
        chunk_size: int = 256,
        delta_init: float = 1.0,
        lambda_init: float = 1e-5,
        method: str = "trf",
        penalty: float = 1e8,
        log_every: int = 100,
        patience: int = 10,
        debug: bool = False,
    ):
        if method not in ("trf", "lm"):
            raise ValueError(
                f"Unknown Refine_fit method: {method!r}. Use 'trf' or 'lm'."
            )
        self.X = X
        self.y = y
        self.params = params
        self.bounds = bounds
        self.model_func = model_func
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_iters = n_iters
        self.chunk_size = chunk_size
        self.delta_init = delta_init  # trust-region radius init, used when method="trf"
        self.lambda_init = lambda_init  # LM damping init, used when method="lm"
        self.method = (
            method  # "trf" (trust-region-radius) or "lm" (Levenberg-Marquardt)
        )
        self.penalty = penalty
        self.n_starts = n_starts
        self.log_every = log_every
        # a fit is only dropped from the active (masked) set once it BOTH satisfies
        # the formal xtol/gtol convergence test AND has failed to improve cost for
        # `patience` consecutive iterations 
        self.patience = patience
        self.debug = debug

    def _cl_scaling_vector(
        self,
        params: torch.Tensor,
        grad: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the scaling vector a la Coleman-Li affine scaling."""
        v = torch.ones_like(params)
        dv = torch.zeros_like(params)
        mask_upper = (grad < 0) & torch.isfinite(upper)[None, :]
        mask_lower = (grad > 0) & torch.isfinite(lower)[None, :]
        v = torch.where(mask_upper, upper[None, :] - params, v)
        dv = torch.where(mask_upper, -torch.ones_like(params), dv)
        mask_lower = (grad > 0) & torch.isfinite(lower)[None, :]
        v = torch.where(mask_lower, params - lower[None, :], v)
        dv = torch.where(mask_lower, torch.ones_like(params), dv)
        return v.clamp_min(1e-12), dv

    def _svd_ridge_step(
        self,
        V: torch.Tensor,
        s: torch.Tensor,
        uf: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Ridge step form the SVD of the augmented and scaled Jacobian.

        Args:
            V: Right singular vectors of the augmented Jacobian (shape: [batch, n_params, n_params])
            s: Singular values of the augmented Jacobian (shape: [batch, n_params])
            uf: Product of U^T and the augmented residual vector (shape: [batch, n_params])
            alpha: Regularisation parameter (shape: [batch]). alpha=0 gives the minimum Gauss-Newton step.

        Returns:
            torch.Tensor: The parameter update step (shape: [batch, n_params])
        """
        denom = s**2 + alpha[:, None]
        coeffs = (s * uf) / denom.clamp_min(1e-300)
        return torch.einsum("bij,bj->bi", V, coeffs)  # same as V @ coeffs

    def _step_size_to_bound(
        self,
        params: torch.Tensor,
        step: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        theta: float = 0.995,
    ) -> torch.Tensor:
        """Calculate the maximum step size to keep params within bounds."""

        eps = 1e-12
        inf = torch.full_like(step, float("inf"))
        pos = step > eps
        neg = step < -eps
        alpha_pos = torch.where(
            pos, (upper[None, :] - params) / step.clamp_min(eps), inf
        )
        alpha_neg = torch.where(
            neg, (lower[None, :] - params) / step.clamp_max(-eps), inf
        )

        alpha = torch.clamp(
            torch.minimum(alpha_pos, alpha_neg).amin(dim=1), min=0.0, max=1.0
        )
        return torch.where(
            alpha < 1.0, alpha * theta, alpha
        )  # scale down if hitting the boundary

    def _solve_lsq_trust_region(
        self,
        V: torch.Tensor,
        s: torch.Tensor,
        uf: torch.Tensor,
        delta: torch.Tensor,
        max_iter: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the least squares problem with trust region constraint.

        Args:
            V: Right singular vectors of the augmented Jacobian (shape: [batch, n_params, n_params])
            s: Singular values of the augmented Jacobian (shape: [batch, n_params])
            uf: Product of U^T and the augmented residual vector (shape: [batch, n_params])
            delta: Trust region radius (shape: [batch])
            max_iter: Maximum number of iterations for the root-finding algorithm.

        Returns:
            delta_q: The parameter update step (shape: [batch, n_params])
            alpha: The regularisation parameter (shape: [batch])
            bound_hit: Boolean tensor indicating if the trust region boundary was hit (shape: [batch])
        """
        eps = 1e-12
        zeros = torch.zeros_like(delta)
        delta_q_gn = self._svd_ridge_step(V, s, uf, zeros)
        gn_norm = torch.norm(delta_q_gn, dim=1)
        rank_eps = torch.finfo(s.dtype).eps * s.shape[1] * s[:, 0].clamp_min(1e-300)
        full_rank = s[:, 0] > rank_eps
        within_region = full_rank & (gn_norm <= delta)

        suf = s * uf
        suf_norm = torch.norm(suf, dim=1)
        alpha_upper = suf_norm / delta.clamp_min(eps)

        def phi_dphi(alpha):
            denom = s**2 + alpha[:, None]
            p_norm = torch.norm(suf / denom, dim=1)
            phi = p_norm - delta
            dphi = -((suf**2 / denom**3).sum(dim=1) / p_norm.clamp_min(eps))
            return phi, dphi

        phi0, dphi0 = phi_dphi(zeros)
        alpha_lower = torch.where(
            full_rank, (-phi0 / dphi0.clamp(max=-eps)).clamp_min(0.0), zeros
        )

        def restart():
            return torch.maximum(
                0.001 * alpha_upper, (alpha_lower * alpha_upper).clamp_min(0.0).sqrt()
            )

        alpha = torch.where(within_region, zeros, restart())
        for _ in range(max_iter):
            out_of_bracket = (alpha < alpha_lower) | (alpha > alpha_upper)
            alpha = torch.where(out_of_bracket, restart(), alpha)
            phi, dphi = phi_dphi(alpha)
            alpha_upper = torch.where((phi < 0) & ~within_region, alpha, alpha_upper)
            ratio = phi / dphi.clamp(max=-eps)
            alpha_lower = torch.where(
                ~within_region, torch.maximum(alpha_lower, alpha - ratio), alpha_lower
            )
            alpha = torch.where(
                within_region,
                alpha,
                alpha - (phi + delta) * ratio / delta.clamp_min(eps),
            )

        delta_q_reg = self._svd_ridge_step(V, s, uf, alpha)
        reg_norm = torch.norm(delta_q_reg, dim=1)
        delta_q_reg = delta_q_reg * (delta / reg_norm.clamp_min(eps))[:, None]

        delta_q = torch.where(within_region[:, None], delta_q_gn, delta_q_reg)
        alpha = torch.where(within_region, zeros, alpha)
        bound_hit = ~within_region
        return delta_q, alpha, bound_hit

    def _check_convergence(
        self,
        cost: torch.Tensor,
        cost_new: torch.Tensor,
        ratio: torch.Tensor,
        p_new: torch.Tensor,
        step: torch.Tensor,
        scaled_g: torch.Tensor,
        improved: torch.Tensor,
        xtol: float = 1e-8,
        ftol: float = 1e-8,
        gtol: float = 1e-8,
    ):
        """Check the convergence criteria for TRF optimisation.

        Args:
            cost: Current cost (shape: [batch])
            cost_new: New cost after the step (shape: [batch])
            ratio: Ratio of actual to predicted reduction (shape: [batch])
            p_new: New parameters after the step (shape: [batch, n_params])
            step: Step taken in parameter space (shape: [batch, n_params])
            scaled_g: Scaled gradient vector (shape: [batch, n_params])
            improved: Boolean tensor indicating if the cost improved (shape: [batch])
        Returns:
            torch.Tensor: boolean tensor indicating which fits have converged
        """
        step_norm = torch.norm(step, dim=1)
        p_norm = torch.norm(p_new, dim=1)
        xtol_converged = step_norm < xtol * (xtol + p_norm)

        actual_reduction = cost - cost_new
        ftol_converged = (actual_reduction < ftol * cost.clamp_min(1e-300)) & (
            ratio > 0.25
        )

        gnorm = torch.max(torch.abs(scaled_g), dim=1).values
        gtol_converged = gnorm < gtol

        return gtol_converged | (improved & (xtol_converged | ftol_converged))

    def _check_convergence_lm(
        self,
        p_new: torch.Tensor,
        step: torch.Tensor,
        scaled_g: torch.Tensor,
        xtol: float = 1e-6,
        gtol: float = 1e-8,
    ) -> torch.Tensor:
        """Convergence check for method="lm" (no trust-region ratio available here).

        Only used to decide when to stop iterating early (see `fit`) -- it does not
        gate whether a step is accepted, which is `improved` alone.
        """
        p_norm = torch.norm(p_new, dim=1)
        step_norm = torch.norm(step, dim=1)
        p_converged = (step_norm / (p_norm + xtol)) < xtol

        gnorm = torch.max(torch.abs(scaled_g), dim=1).values
        g_converged = gnorm < gtol

        return p_converged | g_converged

    def fit(self) -> torch.Tensor:
        """Fit the model to the data using TRF (trust-region) or LM optimisation.

        Which algorithm runs is controlled by `self.method` ("trf" or "lm"). Both
        share the same Jacobian/Coleman-Li-scaling/SVD machinery and only differ in
        how the regularised step is solved for and how the per-fit hyperparameter
        (`radius` -- a trust-region radius for "trf", a damping factor for "lm") is
        updated each iteration.
        """

        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module="torch",
        )
        n_fits, n_params = self.params.shape
        device, dtype = self.params.device, self.params.dtype

        params = self.params.detach().clone()
        # separate out the lower and upper bounds for the parameters
        lower, upper = vectorise_bounds(self.bounds)
        lower = lower.to(device=device, dtype=dtype)
        upper = upper.to(device=device, dtype=dtype)

        radius_init = self.delta_init if self.method == "trf" else self.lambda_init
        radius_all = torch.full((n_fits,), radius_init, device=device, dtype=dtype)
        residual_func = functools.partial(
            calculate_residual,
            model_func=self.model_func,
            penalty=self.penalty,
            bounds=self.bounds,
            optimiser="trf",
        )

        # declare the Jacobian and residual functions
        jac_func = torch.func.vmap(
            torch.func.jacfwd(residual_func, argnums=0), in_dims=(0, None, 0)
        )
        res_func = torch.func.vmap(residual_func, in_dims=(0, None, 0))

        # target is (n_samples, n_fits); the per-chunk loop needs fits-major (row i
        # corresponds to params[i]) but r-squared calculation needs samples-major
        y_samples_major = self.y.repeat_interleave(self.n_starts, dim=1)
        y_expanded = y_samples_major.T.contiguous()  # (n_fits, n_samples)

        # initialise cost and counter for the number of fits improved
        total_cost_before, total_cost_after, n_fits_improved = 0.0, 0.0, 0
        cost_by_iter = torch.zeros((self.n_iters,), device=device, dtype=dtype)

        for start in range(0, n_fits, self.chunk_size):
            print(
                f"[batch-{self.method}] processing fits {start} to {min(start + self.chunk_size, n_fits)} of {n_fits}",
                flush=True,
            )
            end = min(start + self.chunk_size, n_fits)
            p = params[start:end]
            t = y_expanded[start:end]
            radius = radius_all[start:end].clone()

            r = res_func(p, self.X, t)
            cost = (r**2).sum(dim=1)
            cost_before = cost.clone()

            # initialise convergence and stall-patience trackers
            converged = torch.zeros_like(cost, dtype=torch.bool)
            stall_count = torch.zeros_like(cost, dtype=torch.long)
            for it in tqdm(
                range(self.n_iters),
                desc=f"[batch-{self.method}] chunk {start}-{end}",
                unit="iter",
            ):
                # mask out fits that have been dropped from the active set
                active_idx = (~converged).nonzero(as_tuple=True)[0]
                if active_idx.numel() == 0:
                    break
                p_a = p[active_idx]
                t_a = t[active_idx]
                radius_a = radius[active_idx]
                r_a = r[active_idx]
                cost_a = cost[active_idx]
                # keep track of how many times a fit stalls
                stall_a = stall_count[active_idx]

                # calculate the Jacobian, check for NaNs and Infs
                J = torch.nan_to_num(
                    jac_func(p_a, self.X, t_a), nan=0.0, posinf=0.0, neginf=0.0
                )
                # evaluate the gradient vector
                g = torch.einsum("bni,bn->bi", J, r_a)
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                # scale the gradient and the Jacobian
                v, dv = self._cl_scaling_vector(p_a, g, lower, upper)
                d_scale = v.sqrt()
                J_h = (J * d_scale[:, None, :]).double()  # scale the columns of J
                scaled_g = d_scale * g  # scale the parameter-column of the gradient
                if self.method == "trf":
                    diag_h = torch.nan_to_num(
                        (g * dv).double(), nan=0.0, posinf=0.0, neginf=0.0
                    )
                else:
                    # LM needs a clamp to 0
                    diag_h = torch.nan_to_num(
                        (g * v).double(), nan=0.0, posinf=0.0, neginf=0.0
                    ).clamp_min(0.0)
                r_d = torch.nan_to_num(r_a.double(), nan=0.0, posinf=0.0, neginf=0.0)
                # augment J_h on the diagonal with sqrt(diag_h)
                diag_block = torch.diag_embed(diag_h.sqrt())
                J_aug = torch.cat([J_h, diag_block], dim=1)
                f_aug = torch.cat([r_d, torch.zeros_like(diag_h)], dim=1)
                # QR reduction on the augmented Jacobian before SVD
                Q, R = torch.linalg.qr(J_aug, mode="reduced")
                # solve for the parameter update using SVD
                Qtf = torch.einsum("bmi,bm->bi", Q, f_aug)
                U_r, s, Vh = torch.linalg.svd(R, full_matrices=False)
                V = Vh.transpose(-2, -1)
                uf = torch.einsum("bji,bj->bi", U_r, Qtf)

                if self.method == "trf":
                    # trust-region-radius step
                    delta_q, _alpha, bound_hit = self._solve_lsq_trust_region(
                        V, s, uf, radius_a
                    )
                else:
                    # Levenberg-Marquardt step
                    delta_q = self._svd_ridge_step(V, s, uf, radius_a)
                delta_q = torch.nan_to_num(delta_q, nan=0.0, posinf=0.0, neginf=0.0)
                delta_x = (
                    d_scale * delta_q
                )  # map the scaled-space step back to the actual params

                step_frac = self._step_size_to_bound(p_a, -delta_x, lower, upper)
                # get the new parameters
                step = -step_frac[:, None] * delta_x
                p_new = p_a + step
                # evaluate the new residuals and cost for p_new
                r_new = res_func(p_new, self.X, t_a)
                cost_new = (r_new**2).sum(dim=1)
                improved = cost_new < cost_a

                if self.method == "trf":
                    actual_reduction = cost_a - cost_new
                    q_step = step_frac[:, None] * (-delta_q)
                    Jq = torch.einsum("bni,bi->bn", J_aug, q_step)
                    predicted_reduction = -(
                        2.0 * torch.einsum("bi,bi->b", scaled_g, q_step)
                        + (Jq**2).sum(dim=1)
                    )

                    ratio = torch.where(
                        predicted_reduction > 0,
                        actual_reduction / predicted_reduction.clamp_min(1e-300),
                        torch.where(
                            (actual_reduction == 0) & (predicted_reduction == 0),
                            torch.ones_like(cost_a),
                            torch.zeros_like(cost_a),
                        ),
                    )
                    step_h_norm = torch.norm(q_step, dim=1).double()
                    radius_a = torch.where(ratio < 0.25, 0.25 * step_h_norm, radius_a)
                    radius_a = torch.where(
                        (ratio > 0.75) & bound_hit, 2.0 * radius_a, radius_a
                    )
                    radius_a = radius_a.clamp_min(1e-10)
                    iter_converged = self._check_convergence(
                        cost_a, cost_new, ratio, p_new, step, scaled_g, improved
                    )
                else:
                    # LM update rule
                    radius_a = torch.where(
                        improved, radius_a * 0.5, radius_a * 2.0
                    ).clamp(1e-7, 1e7)
                    p_post = torch.where(improved[:, None], p_new, p_a)
                    iter_converged = self._check_convergence_lm(
                        p_post, delta_x, scaled_g
                    )

                # accept only steps that actually reduce cost
                p_a = torch.where(improved[:, None], p_new, p_a)
                r_a = torch.where(improved[:, None], r_new, r_a)
                cost_a = torch.where(improved, cost_new, cost_a)

                # bump the stall counter on any iteration that didn't improve cost,
                # reset it on any that did 
                stall_a = torch.where(
                    improved, torch.zeros_like(stall_a), stall_a + 1
                )
                freeze_a = iter_converged & (stall_a >= self.patience)

                # scatter the active subset's results back into the full-chunk tensors
                p[active_idx] = p_a
                r[active_idx] = r_a
                cost[active_idx] = cost_a
                radius[active_idx] = radius_a
                stall_count[active_idx] = stall_a
                converged[active_idx] = freeze_a

                cost_by_iter[it] += cost.sum()
                if self.debug:
                    if it % 20 == 0 or it == self.n_iters - 1:
                        print(
                            f"  it={it:4d} n_active={active_idx.numel()}/{cost.shape[0]} "
                            f"accept_rate={improved.float().mean().item():.3f} "
                            f"radius_mean={radius_a.mean().item():.3e} radius_max={radius_a.max().item():.3e} "
                            f"cost_finite={torch.isfinite(cost).all().item()} "
                            f"cost_min={cost.min().item():.4e} cost_max={cost.max().item():.4e}",
                            flush=True,
                        )

            n_fits_improved += torch.sum(cost < cost_before).item()
            total_cost_before += cost_before.sum().item()
            total_cost_after += cost.sum().item()
            params[start:end] = p
            radius_all[start:end] = radius
            torch.cuda.synchronize()

        # printing some results
        print(
            f"{self.method.upper()} refine: {n_fits_improved}/{n_fits} parameter-set fits improved",
            flush=True,
        )
        print(
            f"Total SSE before: {total_cost_before:.4f}, after: {total_cost_after:.4f}",
            flush=True,
        )
        print(
            f"Convergence (total SSE across all {n_fits} fits, every {self.log_every} iterations):",
            flush=True,
        )
        for it in range(0, self.n_iters, self.log_every):
            print(f"  iter {it:4d}: SSE = {cost_by_iter[it]:.4f}", flush=True)
        if (self.n_iters - 1) % self.log_every != 0:
            print(
                f"  iter {self.n_iters - 1:4d}: SSE = {cost_by_iter[self.n_iters - 1]:.4f}",
                flush=True,
            )
        # evaluate the R-squared value for each fit and store it in the object for later inspection
        with torch.no_grad():
            z_pred_final = self.model_func(
                self.X,
                params,
                self.bounds,
                optimiser=self.method,
            )
            r2_all = calculate_r2(y_samples_major, z_pred_final)
            self.r2 = r2_all.detach().view(self.y.shape[1], self.n_starts)
            self.cost_by_iter = cost_by_iter.detach()

        return params.clone()
