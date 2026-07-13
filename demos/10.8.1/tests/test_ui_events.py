from mep_routing.ui.events import routing_key_transition


def _state(**overrides):
    state = {
        "machine_angle": 270,
        "auto_placement_mode_idx": 0,
        "routing_strategy_idx": 1,
        "room_start_mode_idx": 0,
        "router_backend_idx": 0,
        "heuristic_mode_idx": 1,
        "graph_type_idx": 2,
        "rotation_mode_idx": 0,
        "weight_mode_idx": 0,
    }
    state.update(overrides)
    return state


COUNTS = {
    "routing_strategy": 3,
    "room_start": 2,
    "router_backend": 2,
    "heuristic": 3,
    "graph": 3,
    "rotation_mode": 2,
    "auto_placement": 3,
    "weight_mode": 2,
}


def test_routing_key_transitions_cycle_solver_state_and_request_required_adapter_work():
    graph = routing_key_transition("cycle_graph", _state(), COUNTS)
    assert graph.state["graph_type_idx"] == 0
    assert graph.rebuild_graph is True
    assert graph.markers == (("Grid:0", (155, 89, 182)),)

    rotation = routing_key_transition("cycle_rotation_mode", _state(), COUNTS)
    assert rotation.state["rotation_mode_idx"] == 1
    assert rotation.apply_rotation_mode is True

    room_start = routing_key_transition("cycle_room_start", _state(room_start_mode_idx=1), COUNTS)
    assert room_start.state["room_start_mode_idx"] == 0
    assert [marker[0] for marker in room_start.markers] == ["Start:0", "Strat:1"]


def test_routing_key_transitions_preserve_auto_placement_policy_and_ignore_unknown_commands():
    rotation = routing_key_transition("rotate_machine", _state(auto_placement_mode_idx=2), COUNTS)
    assert rotation.state["machine_angle"] == 0
    assert rotation.state["auto_placement_mode_idx"] == 0

    weight = routing_key_transition("cycle_weight_mode", _state(auto_placement_mode_idx=2), COUNTS)
    assert weight.state["weight_mode_idx"] == 1
    assert weight.needs_auto_placement is True
    assert weight.refresh_placement_fields is False

    auto = routing_key_transition("toggle_auto_placement", _state(auto_placement_mode_idx=0), COUNTS)
    assert auto.state["auto_placement_mode_idx"] == 2
    assert auto.needs_auto_placement is True
    assert auto.record_history is False
    assert routing_key_transition("unhandled", _state(), COUNTS) is None
