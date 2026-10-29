import re

import flexiznam as flz


def protocol_base_regex(protocol_base):
    """Regex matching protocol_base in recording names.

    "Sphere" and "Spheres" are both used in protocol names (e.g. SphereTubeMotor
    and SpheresTubeMotor), so either spelling matches both.

    Args:
        protocol_base (str): protocol base name.

    Returns:
        str: regex to use with `recordings.name.str.contains`.
    """
    return re.sub(r"Spheres?", "Spheres?", protocol_base)


def normalise_protocol_base(protocol_base):
    """Use the "Spheres" spelling of protocol_base.

    The pipelines compare protocol_base to names like "SpheresTubeMotor", this
    makes "SphereTubeMotor" behave the same way.

    Args:
        protocol_base (str): protocol base name.

    Returns:
        str: protocol base name with "Sphere" replaced by "Spheres".
    """
    return re.sub(r"Spheres?", "Spheres", protocol_base)


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
