import numpy as np

from mep_routing.graphs import adjacency_without_obstacle_bounds, create_runtime_graph


def test_obstacle_bounds_eliminate_graph_vertices_and_edges_numerically():
    runtime = create_runtime_graph(
        np.array(((0, 0), (100, 0), (200, 0)), dtype=np.float32),
        ((0, 1, 100.0, "E"), (1, 2, 100.0, "E")),
    )

    adjacency, blocked_nodes, blocked_edges = adjacency_without_obstacle_bounds(
        runtime, ((80, -10, 120, 10),), protected_nodes=(0, 2),
    )

    assert blocked_nodes == 1
    assert blocked_edges == 2
    assert adjacency == {0: [], 1: [], 2: []}
