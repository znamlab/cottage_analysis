"""`common_utils.iterate_fit`: a restart that does not converge is skipped.

`scipy.optimize.curve_fit` raises `RuntimeError` when it runs out of function
evaluations. One such restart used to abort the whole fit, and with it a full session of
depth fits (PZAG3.4f_S20220504, `_notrunning` 5-fold fit, Nov 2026).
"""

import warnings

import numpy as np
import pytest

from cottage_analysis.analysis import common_utils

LOWER = [0.0, -3.0, 0.1, -1.0]
UPPER = [5.0, 3.0, 3.0, 1.0]
NITER = 6


def gaussian(x, amp, x0, sigma, offset):
    return amp * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 80)
    y = gaussian(x, 2.0, 0.5, 0.6, 0.1) + rng.normal(0, 0.05, x.size)
    return x, y


def _spy_curve_fit(monkeypatch, fail_on=()):
    """Wrap `curve_fit` as seen by `iterate_fit`: record every call and raise
    `RuntimeError` on the calls whose index is in `fail_on`."""
    real_curve_fit = common_utils.curve_fit
    calls = []

    def spy(func, X, y, **kwargs):
        i_call = len(calls)
        call = dict(p0=np.array(kwargs["p0"], dtype=float), popt=None, rsq=None)
        calls.append(call)
        if i_call in fail_on:
            raise RuntimeError(
                "Optimal parameters not found: The maximum number of function "
                "evaluations is exceeded."
            )
        popt, pcov = real_curve_fit(func, X, y, **kwargs)
        call["popt"] = popt
        call["rsq"] = common_utils.calculate_r_squared(y, func(np.array(X), *popt))
        return popt, pcov

    monkeypatch.setattr(common_utils, "curve_fit", spy)
    return calls


def _fit(x, y):
    return common_utils.iterate_fit(gaussian, x, y, LOWER, UPPER, niter=NITER)


def test_all_restarts_converge_no_warning(data, monkeypatch):
    """Nothing changes when every restart converges: best restart, no warning."""
    x, y = data
    calls = _spy_curve_fit(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        popt, rsq = _fit(x, y)
    assert len(calls) == NITER
    i_best = int(np.argmax([c["rsq"] for c in calls]))
    np.testing.assert_array_equal(popt, calls[i_best]["popt"])
    assert rsq == calls[i_best]["rsq"]
    assert rsq > 0.95


def test_failed_restart_is_skipped(data, monkeypatch):
    """The best restart fails: the best of the others is returned, and the other
    restarts start from the same points as in a run without failure."""
    x, y = data
    reference = _spy_curve_fit(monkeypatch)
    _fit(x, y)
    i_best = int(np.argmax([c["rsq"] for c in reference]))
    monkeypatch.undo()

    calls = _spy_curve_fit(monkeypatch, fail_on={i_best})
    with pytest.warns(RuntimeWarning, match="did not converge"):
        popt, rsq = _fit(x, y)

    # every restart is still attempted, from unchanged starting points
    assert len(calls) == NITER
    for call, ref in zip(calls, reference):
        np.testing.assert_array_equal(call["p0"], ref["p0"])
    # the result is the best of the restarts that converged
    others = [c for i, c in enumerate(reference) if i != i_best]
    expected = others[int(np.argmax([c["rsq"] for c in others]))]
    np.testing.assert_array_equal(popt, expected["popt"])
    assert rsq == expected["rsq"]


def test_no_restart_converges_returns_nan(data, monkeypatch):
    x, y = data
    _spy_curve_fit(monkeypatch, fail_on=set(range(NITER)))
    with pytest.warns(RuntimeWarning, match="did not converge"):
        popt, rsq = _fit(x, y)
    assert popt.shape == (len(LOWER),)
    assert np.all(np.isnan(popt))
    assert np.isnan(rsq)
