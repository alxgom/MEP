import numpy as np

from vent_router.routing import (
    line_graph_dir_from_points,
    ordered_small_room_names,
    path_physical_length,
    terminal_node_indices,
    target_heuristic,
)


class Env:
    def __init__(self, nodes):
        self.nodes = np.array(nodes, dtype=float)


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
