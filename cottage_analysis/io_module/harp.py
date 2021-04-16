"""
Functions used to manipulate harp data
"""

import struct
import numpy as np
import mmap
import pandas as pd

# usefull for harp messages:
MESSAGE_TYPE = {1: 'READ', 2: 'WRITE', 3: 'EVENT', 9: 'READ_ERROR', 10: 'WRITE_ERROR'}
PAYLOAD_TYPE = {0: 'isUnsigned', 128: 'isSigned', 64: 'isFloat', 16: 'Timestamp', 1: 'U8', 129: 'S8', 2: 'U16',
                130: 'S16', 4: 'U32', 132: 'S32', 8: 'U64', 136: 'S64', 68: 'Float', 17: 'TimestampedU8',
                145: 'TimestampedS8', 18: 'TimestampedU16', 146: 'TimestampedS16', 20: 'TimestampedU32',
                148: 'TimestampedS32', 24: 'TimestampedU64', 152: 'TimestampedS64', 84: 'TimestampedFloat'}
_PAYLOAD_STRUCT = {1: 'B',  # - T U8 : Unsigned 8 bits
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
_PAYLOAD_STRUCT = {k: '<' + v for k, v in _PAYLOAD_STRUCT.items()}


def _validate_arguments(valid_addresses, valid_msg_type):
    if valid_msg_type is not None:
        if isinstance(valid_msg_type, str) or isinstance(valid_msg_type, int):
            valid_msg_type = valid_msg_type,
        elif not hasattr(valid_msg_type, '__iter__'):
            raise AttributeError('valid_msg_type must be a sequence of int or str')
        valid_msg = []
        TYPE_MESSAGE = {v: k for k, v in MESSAGE_TYPE.items()}
        for msg_type in valid_msg_type:
            if isinstance(msg_type, str):
                valid_msg.append(TYPE_MESSAGE[msg_type.upper()])
            elif isinstance(msg_type, int):
                assert msg_type in MESSAGE_TYPE.keys()
                valid_msg.append(msg_type)
            else:
                raise AttributeError('valid_msg_type must be in %s' % TYPE_MESSAGE.keys())
        valid_msg_type = valid_msg

    if valid_addresses is not None:
        if not hasattr(valid_addresses, '__contains__'):
            valid_addresses = int(valid_addresses),
    return valid_addresses, valid_msg_type


def read_message(path_to_file, verbose=True, valid_addresses=None, valid_msg_type=None):
    """Read binary file containing harp messages

    if verbose is True, display some progress
    valid_addresses can be specified to return only messages with these addresses and ignore the rest
    msg_type
    """

    valid_addresses, valid_msg_type = _validate_arguments(valid_addresses, valid_msg_type)
    all_msgs = []

    with open(path_to_file, "rb") as f:
        mmap_file = mmap.mmap(f.fileno(), 0, mmap.PROT_WRITE)

    if verbose:
        logger = Logger(len(mmap_file))

    with mmap_file as binary_file:
        msg_start = binary_file.read(5)
        while msg_start:
            msg_type, length, address, port, payload_type = struct.unpack('BBBBB', msg_start)

            # skip irrelevant messages
            read_this_message = True
            if (valid_addresses is not None) and (address not in valid_addresses):
                read_this_message = False
            if (valid_msg_type is not None) and (msg_type not in valid_msg_type):
                read_this_message = False
            if not read_this_message:
                binary_file.seek(length - 3, 1)
                if verbose:
                    logger.log(byte_read=len(msg_start) + length - 3, which='skipped')
                # read the next msg_start
                msg_start = binary_file.read(5)
                continue

            # for good messages make an output dictionary and read the rest
            msg = dict(msg_type=MESSAGE_TYPE[msg_type], length=length, address=address, port=port,
                       payload_type=payload_type)
            if length == 255:
                # some payload might be big. Then the length is spread in 2 more bits.
                # see harp protocol
                raise NotImplementedError()

            msg_end = binary_file.read(length - 3)  # ignore the fields I have already read

            msg.update(unpack_payload(msg_end, payload_type))
            msg['calculated_checksum'] = calculate_checksum(msg_start + msg_end[:-1])
            all_msgs.append(msg)

            # read the next message start
            msg_start = binary_file.read(5)
            if verbose:
                logger.log(byte_read=len(msg_start + msg_end), which='read')
        if verbose:
            logger.close()
        return pd.DataFrame(all_msgs)


def calculate_checksum(message):
    chksm = 0
    for i in message:
        chksm += i
    return chksm & 255


def unpack_payload(msg_end, payload_type):
    """Unpack the end of a harp message

    The variable lenght part of the message contains the payload and an extra byte for the checksum.
    This function unpack this into a dictionary
    """
    # find how many data element I have
    payload_struct = _PAYLOAD_STRUCT[payload_type]
    data_size = struct.calcsize(payload_struct[-1])
    offset = 6 if PAYLOAD_TYPE[payload_type].startswith('Timestamp') else 0
    num_elements = (len(msg_end[:-1]) - offset) / data_size
    assert num_elements.is_integer()
    # unpack and put in a dictionary
    full_struct_fmt = payload_struct[:-1] + payload_struct[-1] * int(num_elements) + 'B'
    payload = struct.unpack(full_struct_fmt, msg_end)
    out_dict = {}
    if PAYLOAD_TYPE[payload_type].startswith('Timestamp'):
        out_dict['inner_timestamp_part_s'] = payload[0]
        out_dict['inner_timestamp_part_us'] = np.int32(payload[1]) * 32  # the data is stored in int16 and is us/32.
        out_dict['timestamp_s'] = out_dict['inner_timestamp_part_s'] + np.float64(
            out_dict['inner_timestamp_part_us']) * 1e-6
        if len(payload) > 3:
            out_dict['data'] = payload[2:-1]
    else:
        out_dict['data'] = payload[0]
    out_dict['checksum'] = payload[-1]
    return out_dict


class Logger(object):

    def __init__(self, file_size, log_types=('read', 'skipped'), print_every_n_updates=1000):
        self.nbytes = 0
        self.file_size = file_size
        self.what = {k: 0 for k in log_types}
        self.last_msg_length = 0
        self.print_every_n_updates = print_every_n_updates
        print('Start reading...')

    def log(self, byte_read, which):
        self.nbytes += byte_read
        self.what[which] += 1
        if self.what[which] % self.print_every_n_updates:
            erase_line = ('\b' * self.last_msg_length)
            text_msg = '%d messages %s, %d%% ...' % (self.what[which], which, self.nbytes / self.file_size * 100)
            self.last_msg_length = len(text_msg)
            print(erase_line + text_msg, end="", flush=True)

    def close(self):
        erase_line = ('\b' * self.last_msg_length)
        msg = ', '.join('%d messages %s' % (num, which) for which, num in self.what.items())
        print(erase_line + msg)
