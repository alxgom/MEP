from mep_routing.app import MachinePose, WorkbenchState


def test_installation_poses_and_results_are_independent():
    state = WorkbenchState.create(("sal", "cli"))
    state.installation("sal").pose.set((100, 200), 90)
    state.installation("cli").pose.set((300, 400), 180)

    assert state.installation("sal").pose.as_tuple() == (100, 200, 90)
    assert state.installation("cli").pose.as_tuple() == (300, 400, 180)

    state.replace_runs({"sal": "sal result", "cli": "cli result"})
    assert state.runs == {"sal": "sal result", "cli": "cli result"}

    state.replace_runs({"cli": "next cli result"})
    assert state.installation("sal").result is None
    assert state.runs == {"cli": "next cli result"}


def test_machine_pose_rotation_is_normalized():
    pose = MachinePose(angle_degrees=350)
    assert pose.rotate(20) == 10
