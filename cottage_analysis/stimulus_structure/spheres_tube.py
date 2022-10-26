

def stimulus_structure(param_df, frame_times):
    if len(param_df) != len(frame_times):
        raise IOError('Maybe we have a problem.')
