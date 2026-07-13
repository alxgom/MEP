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
