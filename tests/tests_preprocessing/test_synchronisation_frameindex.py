"""Regression test for non-zero FrameIndex[0] in synchronisation.generate_vs_df."""

import numpy as np
import pandas as pd

from cottage_analysis.preprocessing import synchronisation


def test_generate_vs_df_nonzero_frameindex(monkeypatch):
    """Row indices from find_monitor_frames must map to FrameIndex before merging.

    When Bonsai starts FrameIndex > 0 (e.g. 400), merging monitor_frames_df.closest_frame
    (a 0-based row index into frame_log) directly onto frame_log.FrameIndex shifts
    every stimulus and locomotion sample by FrameIndex[0] frames (or yields NaNs).
    """
    n_frames = 500
    frame_offset = 400
    frame_indices = np.arange(frame_offset, frame_offset + n_frames, dtype=int)
    harp_times = np.arange(n_frames, dtype=float) / 144.0

    dummy_rec = pd.Series({"name": "R_test", "protocol": "SpheresPermTubeReward"})
    monitor_frames_df = pd.DataFrame(
        {
            "closest_frame": np.arange(n_frames, dtype=float),
            "onset_time": harp_times,
        }
    )
    frame_log = pd.DataFrame(
        {
            "FrameIndex": frame_indices,
            "HarpTime": harp_times,
            "MouseZ": np.arange(n_frames, dtype=float)
            * 100.0,  # cm -> m in generate_vs_df
            "EyeZ": np.zeros(n_frames, dtype=float),
        }
    )
    param_log = pd.DataFrame(
        {
            "Frameindex": frame_indices,
            "HarpTime": harp_times,
            "Depth": np.arange(n_frames, dtype=float),
        }
    )

    monkeypatch.setattr(
        synchronisation, "get_str_or_recording", lambda rec, flz: dummy_rec
    )
    monkeypatch.setattr(
        synchronisation, "find_monitor_frames", lambda **kw: monitor_frames_df.copy()
    )
    monkeypatch.setattr(synchronisation, "get_harp_dataset", lambda **kw: object())
    monkeypatch.setattr(
        synchronisation, "get_frame_log", lambda *a, **kw: frame_log.copy()
    )
    monkeypatch.setattr(
        synchronisation, "get_param_log", lambda *a, **kw: param_log.copy()
    )

    vs_df = synchronisation.generate_vs_df(
        recording=dummy_rec,
        photodiode_protocol=5,
        flexilims_session=object(),
        protocol_base="SpheresPermTubeReward",
    )

    assert len(vs_df) == n_frames
    assert np.array_equal(vs_df["monitor_frame"].to_numpy(), frame_indices)
    assert np.allclose(vs_df["mouse_z"].to_numpy(), np.arange(n_frames, dtype=float))
    assert np.allclose(vs_df["Depth"].to_numpy(), np.arange(n_frames, dtype=float))
