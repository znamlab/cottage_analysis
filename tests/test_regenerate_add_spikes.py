import pandas as pd

from cottage_analysis.analysis.spheres import spheres


def test_regenerate_passes_add_spikes(monkeypatch):
    seen = []
    recs = pd.DataFrame({"name": ["S_R1_SpheresPermTubeReward_multidepth"]})
    monkeypatch.setattr(spheres.flz, "get_entity", lambda **kw: {"id": "s"})
    monkeypatch.setattr(spheres.flz, "get_entities", lambda **kw: recs)

    def fake(*args, add_spikes=False, **kwargs):
        seen.append(add_spikes)
        imaging_df = pd.DataFrame({"x": [0]})
        vs_df = pd.DataFrame({"OriginalSize": [0.087]})
        return vs_df, imaging_df, None, None, None, None

    monkeypatch.setattr(spheres, "_process_single_recording_for_session", fake)
    for flag in (False, True):
        spheres.regenerate_frames_all_recordings(
            "S",
            flexilims_session=object(),
            is_multidepth=True,
            do_regenerate_frames=False,
            add_spikes=flag,
        )
    assert seen == [False, True]
