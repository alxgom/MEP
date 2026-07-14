import numpy as np

from mep_routing.routing import (
    line_graph_dir_from_points,
    ordered_small_room_names,
    path_physical_length,
    run_super_sink_line_graph_search,
    run_super_sink_state_astar,
    terminal_node_indices,
    target_heuristic,
)


class Env:
    def __init__(self, nodes, adj=None):
        self.nodes = np.array(nodes, dtype=float)
        self.adj = adj or {}


def test_line_graph_dir_from_points_uses_dominant_axis():
    env = Env([(0, 0), (10, 2), (2, -10)])

    assert line_graph_dir_from_points(env, 0, 1) == "E"
    assert line_graph_dir_from_points(env, 0, 2) == "S"


def test_path_physical_length_sums_euclidean_segments():
    env = Env([(0, 0), (3, 4), (3, 14)])

    assert path_physical_length(env, [0, 1, 2]) == 15.0


def test_target_heuristic_uses_manhattan_distance_without_turns():
    env = Env([(0, 0), (10, 0), (0, 8)])
    targets = [{"node_idx": 1}, {"node_idx": 2}]

    assert target_heuristic(
        env,
        0,
        "E",
        targets,
        bend_cost=100,
        heuristic_mode=1,
        machine_center=(0, 0),
        estimate_turns_fn=lambda _p, _incoming, _t: 99,
    ) == 8.0


def test_target_heuristic_adds_turn_cost_in_bend_aware_mode():
    env = Env([(0, 0), (10, 0)])

    assert target_heuristic(
        env,
        0,
        "N",
        [{"node_idx": 1}],
        bend_cost=100,
        heuristic_mode=0,
        machine_center=(0, 0),
        estimate_turns_fn=lambda _p, _incoming, _t: 2,
    ) == 210.0


def test_target_heuristic_machine_ring_mode_uses_distance_outside_target_radius():
    env = Env([(100, 0), (30, 0), (0, 40)])
    targets = [{"node_idx": 1}, {"node_idx": 2}]

    assert target_heuristic(
        env,
        0,
        "E",
        targets,
        bend_cost=100,
        heuristic_mode=2,
        machine_center=(0, 0),
        estimate_turns_fn=lambda _p, _incoming, _t: 0,
    ) == 60.0


def test_target_heuristic_disabled_mode_and_invalid_node_return_zero():
    env = Env([(0, 0), (10, 0)])

    assert target_heuristic(env, -1, None, [{"node_idx": 1}], 100, 1, (0, 0), lambda *_: 1) == 0.0
    assert target_heuristic(env, 0, None, [{"node_idx": 1}], 100, 3, (0, 0), lambda *_: 1) == 0.0


def test_terminal_node_indices_queries_kd_and_preserves_shaft_node():
    class FakeKd:
        def query(self, point):
            return 0.0, {"Bath": 4, "Kitchen": 7}[point]

    assert terminal_node_indices({"Bath": "Bath", "Kitchen": "Kitchen"}, 2, FakeKd()) == {
        "Shaft": 2,
        "Bath": 4,
        "Kitchen": 7,
    }
    assert terminal_node_indices({}, 3, FakeKd(), shaft_route_name="Core") == {"Core": 3}


def test_ordered_small_room_names_filters_and_sorts_by_machine_distance():
    terminals = {
        "Kitchen": (1.0, 1.0),
        "Bathroom 2": (100.0, 0.0),
        "Bedroom": (1.0, 0.0),
        "Toilet": (10.0, 0.0),
        "Washroom": (50.0, 0.0),
    }

    assert ordered_small_room_names(terminals, machine_center=(0.0, 0.0)) == [
        "Toilet",
        "Washroom",
        "Bathroom 2",
    ]


def _search_env():
    return Env(
        [(0, 0), (10, 0), (20, 0), (0, 1)],
        {
            0: [(1, 10.0, "E"), (3, 1.0, "N")],
            1: [(0, 10.0, "W"), (2, 10.0, "E")],
            2: [(1, 10.0, "W"), (3, 21.0, "S")],
            3: [(0, 1.0, "S"), (2, 21.0, "E")],
        },
    )


def _target_specs():
    return {
        "large": [{"node_idx": 2, "in_dir": "E", "out_dir": "W", "pin": "large"}],
    }


def test_state_expanded_astar_selects_the_lower_turn_cost_path_to_a_directed_pin():
    path, length, pin_name, target = run_super_sink_state_astar(
        _search_env(),
        start_node_indices=0,
        target_pin_names=["large"],
        pin_node_map=_target_specs(),
        bend_cost=20.0,
        heuristic_mode=1,
        estimate_turns_fn=lambda *_: 0,
    )

    assert path == [0, 1, 2]
    assert length == 20.0
    assert pin_name == "large"
    assert target == _target_specs()["large"][0]


def test_line_graph_astar_selects_the_lower_turn_cost_path_to_a_directed_pin():
    path, length, pin_name, target = run_super_sink_line_graph_search(
        _search_env(),
        start_node_indices=[0],
        target_pin_names=["large"],
        pin_node_map=_target_specs(),
        bend_cost=20.0,
        heuristic_mode=1,
        estimate_turns_fn=lambda *_: 0,
    )

    assert path == [0, 1, 2]
    assert length == 20.0
    assert pin_name == "large"
    assert target == _target_specs()["large"][0]
