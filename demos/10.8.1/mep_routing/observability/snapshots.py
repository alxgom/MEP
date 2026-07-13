"""Serializable session snapshot construction."""


def solution_snapshot(state, kpis, status, total_nodes):
    snapshot = dict(state)
    snapshot["preferred_terminal_points_by_room"] = {
        room_name: [tuple(point) for point in points]
        for room_name, points in state["preferred_terminal_points_by_room"].items()
    }
    snapshot["preferred_terminal_areas"] = [
        {"room": area["room"], "bounds": tuple(area["bounds"])}
        for area in state["preferred_terminal_areas"]
    ]
    snapshot.update(status=status, total_nodes=int(total_nodes), kpis=kpis)
    return snapshot


def restored_snapshot_state(snapshot, default_bend_weight, default_crossing_multiplier):
    """Normalize persisted session values for application back into the live app."""
    state = dict(snapshot)
    state["bend_weight"] = snapshot.get("bend_weight", default_bend_weight)
    state["crossing_penalty_multiplier"] = snapshot.get(
        "crossing_penalty_multiplier", default_crossing_multiplier
    )
    state["rotation_mode_idx"] = snapshot.get("rotation_mode_idx", 0)
    state["preferred_terminal_points_by_room"] = {
        room_name: [tuple(point) for point in points]
        for room_name, points in snapshot["preferred_terminal_points_by_room"].items()
    }
    state["preferred_terminal_areas"] = [
        {"room": area["room"], "bounds": tuple(area["bounds"])}
        for area in snapshot.get("preferred_terminal_areas", [])
    ]
    return state
