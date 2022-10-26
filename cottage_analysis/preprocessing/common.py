"""
Common preprocessing steps
"""
import flexiznam as flm
from cottage_analysis.io_module import harp


def preprocess_harp(flexilims_name, flexilims_session,  reward_port=36,
                    wheel_diameter=harp.WHEEL_DIAMETER, ecoder_cpr=harp.ENCODER_CPR,
                    inverse_rotary=True, di_names=('lick_detection', 'onix_clock',
                                                   'di2_encoder_initial_state')):
    """Read messages and save processed output"""
    kwargs = dict(reward_port=reward_port,
                  wheel_diameter=wheel_diameter,
                  ecoder_cpr=ecoder_cpr,
                  inverse_rotary=inverse_rotary,
                  di_names=di_names)

    harp_ds = flm.get_entity(datatype='Dataset', name=flexilims_name,
                             flexilims_session=flexilims_session)
    if harp_ds is None:
        raise IOError('Cannot find dataset named `%s` on flexilims.' % flexilims_name)

    harp_df = harp.load_harp(harp_ds.raw_path, **kwargs)
    return harp_df
