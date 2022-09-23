"""Test by loading pilote data"""
from cottage_analysis.io_module import onix
from cottage_analysis.ephys import preprocessing

PROJECT = 'blota_onix_pilote'
MOUSE = 'BRAC6692.4a'
SESSION = 'S20220831'
ONIX_RECORDING = 'R163359'
VIS_STIM_RECORDING = 'R163332_SpheresPermTubeReward'


def test_load_onix_recording():
    out = onix.load_onix_recording(PROJECT, MOUSE, SESSION,
                                   vis_stim_recording=VIS_STIM_RECORDING,
                                   onix_recording=ONIX_RECORDING, allow_reload=False)
    for what in ['harp_message', 'breakout_data', 'rhd2164_data', 'ts4131_data']:
        assert what in out
    assert 'rotary_meter' in out['harp_message']


def test_sync_onix():
    out = onix.load_onix_recording(PROJECT, MOUSE, SESSION,
                                   vis_stim_recording=VIS_STIM_RECORDING,
                                   onix_recording=ONIX_RECORDING, allow_reload=True)
    out = preprocessing.preprocess_exp(out, plot_dir=None)
    assert 'harp2onix' in out
    assert 'hf_frames' in out