import numpy as np

from cottage_analysis.analysis.roi_location import spatial_gradient


def test_spatial_gradient_recovers_known_gradient():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 500, (2, 300))
    values = 0.02 * x - 0.01 * y + rng.normal(0, 0.5, 300)
    g = spatial_gradient(x, y, values, n_perm=500)
    assert g["n"] == 300
    np.testing.assert_allclose([g["slope_x"], g["slope_y"]], [0.02, -0.01], atol=2e-3)
    np.testing.assert_allclose(
        g["direction"], np.degrees(np.arctan2(-0.01, 0.02)), atol=5
    )
    assert g["pval_f"] < 1e-10
    assert g["pval_perm"] == 1 / 501


def test_spatial_gradient_null_and_nan_handling():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 500, (2, 300))
    values = rng.normal(0, 1, 300)
    values[:10] = np.nan
    g = spatial_gradient(x, y, values, n_perm=500)
    assert g["n"] == 290
    assert g["pval_f"] > 0.01 and g["pval_perm"] > 0.01
