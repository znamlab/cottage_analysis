import pytest
import os
from io_module import harp

ROOT_DIR = "./resources/test_data"
EXAMPLE_HARP = "harp_messages_example.bin"


def test_harp():
    fpath = os.path.join(ROOT_DIR, EXAMPLE_HARP)

    msg_df = harp.read_message(fpath, verbose=False)
    assert msg_df.shape == (5000, 11)
    msg_df = harp.read_message(fpath, verbose=False, valid_addresses=32)
    assert msg_df.shape == (429, 11)
    msg_df = harp.read_message(fpath, verbose=False, valid_addresses=(12, 0))
    assert msg_df.shape == (2, 11)
    msg_df = harp.read_message(fpath, verbose=False, valid_addresses=(12, 44), valid_msg_type=2)
    assert msg_df.shape == (0, 0)
    msg_df = harp.read_message(fpath, verbose=False, valid_msg_type='event')
    assert msg_df.shape == (4895, 11)
    msg_df = harp.read_message(fpath, verbose=False, valid_addresses=(12, 44), valid_msg_type=['write', 'Read'])
    assert msg_df.shape == (2, 11)
    msg_df = harp.read_message(fpath, verbose=False, valid_addresses=(12, 44), valid_msg_type=[1, 3])
    assert msg_df.shape == (4469, 11)

