import flexiznam as flz


def get_str_or_recording(recording, flexilims_session):
    """Get recording entry from flexilims_session if recording is a string.

    Args:
        recording (str or pandas.Series): recording name or recording entry from flexilims.
        flexilims_session (flexilims.Flexilims): flexilims session.

    Returns:
        pandas.Series: recording entry from flexilims_session.
    """

    if recording is None:
        return None
    elif type(recording) == str:
        recording = flz.get_entity(
            datatype="recording",
            name=recording,
            flexilims_session=flexilims_session,
        )
        if recording is None:
            raise ValueError(f"Recording {recording} does not exist.")
        return recording
    else:
        return recording
