import numpy as np

from mep_routing.graphs import EnvView
from mep_routing.placement import compute_dijkstra_distance_field, placement_weights, topological_placement_scores


def _line_env():
    nodes = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [300.0, 0.0],
            [500.0, 0.0],
        ]
    )
    adj = {
        0: [(1, 100.0, "E")],
        1: [(0, 100.0, "W"), (2, 200.0, "E")],
        2: [(1, 200.0, "W")],
        3: [],
    }
    return EnvView(nodes, adj)


def test_compute_dijkstra_distance_field_handles_single_and_unreachable_nodes():
    distances = compute_dijkstra_distance_field(0, _line_env())

    assert distances[0] == 0.0
    assert distances[1] == 100.0
    assert distances[2] == 300.0
    assert distances[3] == 1e9


def test_compute_dijkstra_distance_field_accepts_multiple_start_nodes():
    distances = compute_dijkstra_distance_field([0, 2], _line_env())

    assert distances[0] == 0.0
    assert distances[1] == 100.0
    assert distances[2] == 0.0


def test_placement_weights_match_current_modes():
    assert placement_weights(1)["Shaft"] == 1.0
    assert placement_weights(0)["Shaft"] == 2.5
    assert placement_weights(0)["Kitchen"] == 1.5


def test_topological_placement_scores_combines_reachable_distance_fields():
    scores, fields = topological_placement_scores(
        _line_env(),
        shaft_boundary_nodes=[0],
        terminal_nodes={"Kitchen": 2},
        weights={"Shaft": 2.0, "Kitchen": 0.5},
    )

    assert scores == {
        0: 150.0,
        1: 300.0,
        2: 600.0,
    }
    assert fields["Shaft"][2] == 300.0
    assert fields["Kitchen"][0] == 300.0
