from pathlib import Path
import os
from cottage_analysis.io_module import harp
import flexiznam as flz


def test_harp():
    fpath = Path(__file__).parent / "test_data" / "harp_messages_example.bin"

    msg_df = harp.read_message(str(fpath))
    assert len(msg_df) == 5000
    msg_df = harp.read_message(fpath, valid_addresses=32)
    assert len(msg_df) == 429
    msg_df = harp.read_message(fpath, valid_addresses=(12, 0))
    assert len(msg_df) == 2
    msg_df = harp.read_message(fpath, valid_addresses=(12, 44), valid_msg_type=2)
    assert len(msg_df) == 0
    msg_df = harp.read_message(fpath, valid_msg_type="event")
    assert len(msg_df) == 4895
    msg_df = harp.read_message(
        fpath, valid_addresses=(12, 44), valid_msg_type=["write", "Read"]
    )
    assert len(msg_df) == 2
    msg_df = harp.read_message(fpath, valid_addresses=(12, 44), valid_msg_type=[1, 3])
    assert len(msg_df) == 4469


def test_load_harp():
    proj = "hey2_3d-vision_foodres_20220101"
    raw = flz.get_data_root("raw", project=proj) / proj
    fp = raw / "PZAH6.4b/S20220512/R190248_SpheresPermTubeReward"
    fp /= "PZAH6.4b_S20220512_R190248_SpheresPermTubeReward_harpmessage.bin"
    harp_df = harp.load_harp(fp)
    assert len(harp_df) == 10
    assert len(harp_df["reward_times"]) == 200

    proj = "ccyp_ex-vivo-reg-pilot"
    raw = flz.get_data_root("raw", project=proj) / proj
    fp = raw / "BRBQ77.1f/S20231107/R120856_SFTF/harpmessage.bin"
    harp_df = harp.load_harp(fp)
    assert len(harp_df) == 10
    assert len(harp_df["reward_times"]) == 0


if __name__ == "__main__":
    test_harp()
    test_load_harp()
