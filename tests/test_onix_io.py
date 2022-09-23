"""Test by loading pilote data"""
from cottage_analysis.io_module import onix

PROJECT = 'blota_onix_pilote'
MOUSE = 'BRAC6692.4a'
SESSION = 'S20220831'
ONIX_RECORDING = 'R163359'
VIS_STIM_RECORDING = 'R163332_SpheresPermTubeReward'


def test_load_onix_recording():
    out = onix.load_onix_recording(PROJECT, MOUSE, SESSION,
                                   vis_stim_recording=VIS_STIM_RECORDING,
                                   onix_recording=ONIX_RECORDING, allow_reload=True)
    for what in ['harp_message', 'breakout_data', 'rhd2164_data', 'ts4131_data',
                 'harp2onix']:
        assert what in out
