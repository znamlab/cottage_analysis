from pathlib import Path
import os
from cottage_analysis.io_module import harp

ROOT_DIR = Path("tests_io_module/test_data")
EXAMPLE_HARP = "harp_messages_example.bin"


def test_harp():
    fpath = ROOT_DIR / EXAMPLE_HARP
    msg_df = harp.read_message(str(fpath), verbose=False)
    assert len(msg_df) == 5000
    msg_df = harp.read_message(fpath, verbose=True, valid_addresses=32)
    assert len(msg_df) == 429
    msg_df = harp.read_message(fpath, verbose=False, valid_addresses=(12, 0))
    assert len(msg_df) == 2
    msg_df = harp.read_message(fpath, verbose=False, valid_addresses=(12, 44),
                               valid_msg_type=2)
    assert len(msg_df) == 0
    msg_df = harp.read_message(fpath, verbose=False, valid_msg_type='event')
    assert len(msg_df) == 4895
    msg_df = harp.read_message(fpath, verbose=False, valid_addresses=(12, 44),
                               valid_msg_type=['write', 'Read'])
    assert len(msg_df) == 2
    msg_df = harp.read_message(fpath, verbose=False, valid_addresses=(12, 44),
                               valid_msg_type=[1, 3])
    assert len(msg_df) == 4469

