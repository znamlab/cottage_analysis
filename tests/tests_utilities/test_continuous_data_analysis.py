import matplotlib.pyplot as plt
import numpy as np

import cottage_analysis.utilities.continuous_data_analysis as cda


def test_correlate_with_lag():
    """
    Test that my crosscorrelation with maximum lag returns the same as a full correlation
    on valid part. This does not test the Pearson normalisation
    """
    rng = np.random.default_rng(seed=12)
    x = rng.random(1000)
    y = rng.random(1000)
    for maxlag in [5, 10]:
        for expected_lag in [0, 3, 10]:
            corr_np = np.correlate(x[maxlag + expected_lag:-maxlag + 1],
                                   y[:len(y)-expected_lag], mode='valid')
            corr, lags = cda.crosscorrelation(x, y, maxlag=maxlag,
                                              expected_lag=expected_lag,
                                              normalisation='dot')
            assert all(np.abs(corr_np - corr) < 1e-12)
            # check that the lag is what we expect
            corr, lags = cda.crosscorrelation(x[:-10], x[10:], maxlag=20,
                                              expected_lag=0,
                                              normalisation='dot')
            assert lags[corr.argmax()] == 10
