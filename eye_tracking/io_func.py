import os
import struct
import cv2
import numpy as np
import mmap
import pandas as pd

#usefull for harp messages:
MESSAGE_TYPE = {1: 'READ', 2: 'WRITE', 3: 'EVENT', 9: 'READ_ERROR', 10: 'WRITE_ERROR'}
PAYLOAD_TYPE = {0: 'isUnsigned', 128: 'isSigned', 64: 'isFloat', 16: 'Timestamp', 1: 'U8', 129: 'S8', 2: 'U16',
                130: 'S16', 4: 'U32', 132: 'S32', 8: 'U64', 136: 'S64', 68: 'Float', 17: 'TimestampedU8',
                145: 'TimestampedS8', 18: 'TimestampedU16', 146: 'TimestampedS16', 20: 'TimestampedU32',
                148: 'TimestampedS32', 24: 'TimestampedU64', 152: 'TimestampedS64', 84: 'TimestampedFloat'}
PAYLOAD_STRUCT = {1: 'B',  # - T U8 : Unsigned 8 bits
                  2: 'H',  # T U16 : Unsigned 16 bits
                  4: 'I',  # T U32 : Unsigned 32 bits
                  8: 'Q',  # T U32 : Unsigned 64 bits
                  129: 'b',  # T I8 : Signed 8 bits
                  130: 'h',  # T I16 : Signed 16 bits
                  132: 'i',  # T I32 : Signed 32 bits
                  136: 'l',  # T I64 : Signed 64 bits
                  68: 'f',  # T Float : Single-precision floating-point 32 bits
                  16: 'IH',  # Timestamped<> : Time information only
                  17: 'IHB',  # Timestamped<T> U8 : Timestamped unsigned 8 bits
                  18: 'IHH',  # Timestamped<T> U16 : Timestamped unsigned 16 bits
                  20: 'IHI',  # Timestamped<T> U32 : Timestamped unsigned 32 bits
                  24: 'IHQ',  # Timestamped<T> U64 : Timestamped unsigned 64 bits
                  145: 'IHb',  # Timestamped<T> I8 : Timestamped signed 8 bits
                  146: 'IHh',  # Timestamped<T> I16 : Timestamped signed 16 bits
                  148: 'IHi',  # Timestamped<T> I32 : Timestamped signed 32 bits
                  152: 'IHl',  # Timestamped<T> I64 : Timestamped signed 64 bits
                  84: 'IHf'  # Timestamped<T> Float : Timestamped Single-precision floating-point 32 bits
                  }
# specify endianess explicitly
PAYLOAD_STRUCT = {k: '<' + v for k, v in PAYLOAD_STRUCT.items()}

DEPTH_DICT = {8: np.uint8,
              16: np.uint16}


def load_video(data_folder, camera='right_eye_camera'):
    """Load the video from an eye cam"""
    metadata_file = os.path.join(data_folder, '%s_metadata.txt' % camera)
    assert os.path.isfile(metadata_file)
    metadata = {}
    with open(metadata_file, 'r') as m_raw:
        for line in m_raw:
            if line.strip():
                k, v = line.strip().split(":")
                metadata[k.strip()] = int(v.strip())

    binary_file = os.path.join(data_folder, '%s_data.bin' % camera)
    assert os.path.isfile(binary_file)
    data = np.memmap(binary_file, dtype=DEPTH_DICT[metadata['Depth']], mode='r')
    data = data.reshape((metadata['Height'], metadata['Width'], -1), order='F')
    return data


def write_mp4(target_file, video_array, frame_rate, is_color=False, codec='mp4v', extension='.mp4', overwrite=False):
    """Write an array to a mp4 file

    The array must shape must be (lines/height x columns/width x frames)
    """
    if not target_file.endswith(extension):
        target_file += extension
    if not overwrite:
        assert not os.path.isfile(target_file)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    # writer to `output` in mp4v at 20 fps of the correct size, NOT color
    out = cv2.VideoWriter(target_file, fourcc, frame_rate, (video_array.shape[1], video_array.shape[0]), is_color)
    for frame in range(video_array.shape[2]):
        out.write(video_array[:, :, frame])
    return out


def read_message(path_to_file, verbose=True, address_to_read=None):
    """Read binary file containing harp messages

    if verbose is True, display some progress
    """
    all_msgs = []
    if verbose:
        print('Start reading')
        msg_read = 0
        msg_skipped = 0
        text_msg = ''

    with open(path_to_file, "rb") as f:
        mmap_file = mmap.mmap(f.fileno(), 0, mmap.PROT_WRITE)

    with mmap_file as binary_file:
        # binary_file.seek(0)
        msg_start = binary_file.read(5)
        while msg_start:
            msg_type, length, address, port, payload_type = struct.unpack('BBBBB', msg_start)
            if (address_to_read is not None) and (address not in address_to_read):
                # print(binary_file.tell())
                binary_file.seek(length - 3, 1)
                # print(length)
                # print(binary_file.tell())
                if verbose:
                    msg_skipped += 1
                    if msg_skipped % 1000 == 0:
                        erase_line = ('\b' * len(text_msg))
                        text_msg = 'Skipped %12d messages ...' % msg_skipped
                        print(erase_line + text_msg)
                continue
            assert msg_type in MESSAGE_TYPE
            if length == 255:
                raise NotImplementedError()

            # now read the rest of the message
            assert length > 3
            assert payload_type in PAYLOAD_TYPE
            msg_end = binary_file.read(length - 3)  # ignore the fields I have already read
            # find how many data element I have
            payload_struct = PAYLOAD_STRUCT[payload_type]
            data_size = struct.calcsize(payload_struct[-1])
            offset = 6 if PAYLOAD_TYPE[payload_type].startswith('Timestamp') else 0
            num_elements = (len(msg_end[:-1]) - offset) / data_size
            assert num_elements.is_integer()
            full_struct_fmt = payload_struct[:-1] + payload_struct[-1] * int(num_elements) + 'B'
            payload = struct.unpack(full_struct_fmt, msg_end)
            msg = dict(msg_type=MESSAGE_TYPE[msg_type], length=length, address=address, port=port,
                       payload_type=payload_type)
            if PAYLOAD_TYPE[payload_type].startswith('Timestamp'):
                msg['inner_timestamp_part_s'] = payload[0]
                msg['inner_timestamp_part_us'] = np.int32(payload[1]) * 32  # the data is stored in int16 and is us/32.
                msg['timestamp_s'] = msg['inner_timestamp_part_s'] + msg['inner_timestamp_part_us'] * 1e-6
                if len(payload) > 3:
                    msg['data'] = payload[2:-1]
            else:
                msg['data'] = payload[0]
            msg['checksum'] = payload[-1]
            chksm = 0
            for i in msg_start + msg_end[:-1]:
                chksm += i
            msg['calculated_checksum'] = chksm & 255
            all_msgs.append(msg)

            # read the next message start
            msg_start = binary_file.read(5)
            if verbose:
                msg_read += 1
                if msg_read % 1000 == 0:
                    erase_line = ('\b' * len(text_msg))
                    text_msg = 'Read %12d messages ...' % msg_read
                    print(erase_line + text_msg)
        if verbose:
            print("Packing into dataframe...")
        return pd.DataFrame(all_msgs)


if __name__ == "__main__":
    ROOT_DIR = "../resources/test_data"
    EXAMPLE_FILE = "PZAH4.1c_harpmessage_S20210406_R184923.bin"

    fpath = os.path.join(ROOT_DIR, EXAMPLE_FILE)

    msg = read_message(fpath, verbose=False)
    MOUSE = "PZAH4.1c"
    SESSION = "S20210406"
    RECORDING = "R184923"
    camera = 'right_eye_camera'
    data_folder = os.path.join(ROOT_DIR, MOUSE, SESSION, RECORDING)
    vid = load_video(data_folder=data_folder, camera=camera)
    print('Video loaded')
    codec = 'RGBA'
    extension = '.avi'
    target_file = '_'.join([MOUSE, SESSION, RECORDING, camera]) + extension
    write_mp4(target_file=os.path.join(data_folder, target_file), video_array=vid, frame_rate=120,
              codec=codec, extension=extension)
    print('done')
    import os

