import flexiznam as flz


def get_sessions(
    flexilims_session,
    exclude_sessions=(),
    exclude_openloop=True,
    exclude_pure_closedloop=False,
    v1_only=True,
    trialnum_min=10,
    mouse_list=None,
):
    """
    Get a list of sessions to include.

    Args:
        flexilims_session (str): flexilims session
        exclude_sessions (list, optional): list of sessions to exclude manually.
            Defaults to ().
        exclude_openloop (bool, optional): only include closedloop sessions. Defaults to
            True. To include all sessions, set to False.
        exclude_pure_closedloop (bool, optional): only include openloop sessions. Defaults to
            False. To include all sessions, set to False.
        v1_only (bool, optional): only include V1 sessions. Defaults to True.
        trialnum_min (int, optional): minimum number of trials per depth to include a session.
        mouse_list (list, optional): list of mice to include, if None, include all.
            Default to None.

    Returns:
        list: list of sessions to include

    """
    assert not (
        exclude_openloop and exclude_pure_closedloop
    ), "Both closedloop_only and openloop_only cannot be True"
    session_list = []
    if mouse_list is None:
        mouse_list = flz.get_entities("mouse", flexilims_session=flexilims_session)

    # get children is too slow. It's better to get everything and filter
    project_sessions = flz.get_entities("session", flexilims_session=flexilims_session)
    project_recordings = flz.get_entities(
        "recording", flexilims_session=flexilims_session,
    )
    for mouse_id in mouse_list["id"].values:
        sessions_mouse = project_sessions[project_sessions.origin_id == mouse_id]
        # exclude any sessions which have an "exclude_reason" on flexilims
        if "exclude_reason" in sessions_mouse.columns:
            if not v1_only:
                sessions_mouse = sessions_mouse[
                    (sessions_mouse["exclude_reason"].isna())
                    | (sessions_mouse["exclude_reason"].str.isspace())
                    | (sessions_mouse["exclude_reason"] == "not V1")
                ]
            else:
                sessions_mouse = sessions_mouse[
                    (
                        (sessions_mouse["exclude_reason"].isna())
                        | (sessions_mouse["exclude_reason"].str.isspace())
                    )
                    & (
                        sessions_mouse["closedloop_trials"] / sessions_mouse["ndepths"]
                        > trialnum_min
                    )
                ]
        if len(sessions_mouse) > 0:
            session_list.append(sessions_mouse.name.values.tolist())
    # exclude any sessions from exclude_sessions
    session_list = [session for i in session_list for session in i]
    session_list = [
        session for session in session_list if session not in exclude_sessions
    ]
    keep_sessions = []
    # filter based on closedloop_only and openloop_only
    for session in session_list:
        sess_id = project_sessions.loc[session].id
        recs = project_recordings[project_recordings.origin_id == sess_id]
        open_loop = (recs["protocol"] == "SpheresPermTubeRewardPlayback").any()
        if exclude_openloop and open_loop:
            continue
        if exclude_pure_closedloop and (not open_loop):
            continue
        keep_sessions.append(session)
    return keep_sessions


def get_size_control_sessions(
    flexilims_session,
    exclude_sessions=(),
    v1_only=True,
    mouse_list=None,
    protocol_base="SizeControl",
):
    """
    Get a list of sessions with SizeControl recordings.

    Unlike get_sessions, this does not filter by closedloop_trials/ndepths since
    those columns are specific to the depth-tuning (SpheresPermTubeReward) protocol.

    Args:
        flexilims_session (str): flexilims session
        exclude_sessions (list, optional): list of sessions to exclude manually.
            Defaults to ().
        v1_only (bool, optional): only include V1 sessions. Defaults to True.
        mouse_list (list, optional): list of mice to include, if None, include all.
            Default to None.
        protocol_base (str, optional): substring identifying SizeControl recordings.
            Defaults to "SizeControl".

    Returns:
        list: list of session names with at least one SizeControl recording

    """
    session_list = []
    if mouse_list is None:
        mouse_list = flz.get_entities("mouse", flexilims_session=flexilims_session)

    project_sessions = flz.get_entities("session", flexilims_session=flexilims_session)
    project_recordings = flz.get_entities(
        "recording", flexilims_session=flexilims_session,
    )
    for mouse_id in mouse_list["id"].values:
        sessions_mouse = project_sessions[project_sessions.origin_id == mouse_id]
        if "exclude_reason" in sessions_mouse.columns:
            # sessions are tagged exclude_reason="size control" precisely because
            # they're SizeControl sessions excluded from the depth-tuning figures;
            # that reason is expected here and must not be treated as "not V1".
            is_blank = (sessions_mouse["exclude_reason"].isna()) | (
                sessions_mouse["exclude_reason"].str.isspace()
            )
            is_size_control_reason = sessions_mouse["exclude_reason"].str.lower().str.replace(
                " ", ""
            ) == protocol_base.lower().replace(" ", "")
            if not v1_only:
                sessions_mouse = sessions_mouse[
                    is_blank
                    | is_size_control_reason
                    | (sessions_mouse["exclude_reason"] == "not V1")
                ]
            else:
                sessions_mouse = sessions_mouse[is_blank | is_size_control_reason]
        if len(sessions_mouse) > 0:
            session_list.append(sessions_mouse.name.values.tolist())
    session_list = [session for i in session_list for session in i]
    session_list = [
        session for session in session_list if session not in exclude_sessions
    ]

    keep_sessions = []
    for session in session_list:
        sess_id = project_sessions.loc[session].id
        recs = project_recordings[project_recordings.origin_id == sess_id]
        has_size_control = recs["protocol"].str.contains(protocol_base, na=False).any()
        if has_size_control:
            keep_sessions.append(session)
    return keep_sessions
