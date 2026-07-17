import numpy as np

from mep_routing.installations.cli.solver_backends import solve_line_graph_mst


def _connect(adjacency, left, right, distance, direction):
    adjacency[left].append((right, distance, direction))
    opposite = {"E": "W", "W": "E", "N": "S", "S": "N"}[direction]
    adjacency[right].append((left, distance, opposite))


def test_line_graph_mst_applies_cli_bend_weight():
    nodes = np.asarray(((0, 0), (1, 0), (1, 1), (2, 1), (1, 0.5)), dtype=float)
    adjacency = {index: [] for index in range(len(nodes))}
    _connect(adjacency, 0, 1, 1, "E")
    _connect(adjacency, 1, 2, 1, "S")
    _connect(adjacency, 2, 3, 1, "E")
    _connect(adjacency, 0, 4, 2.5, "E")
    _connect(adjacency, 4, 3, 2.5, "E")

    shortest = solve_line_graph_mst(nodes, adjacency, (0, 3), bend_penalty=0)
    straightest = solve_line_graph_mst(nodes, adjacency, (0, 3), bend_penalty=10)

    assert set(shortest.tree_edges) == {(0, 1), (1, 2), (2, 3)}
    assert set(straightest.tree_edges) == {(0, 4), (3, 4)}


def test_line_graph_mst_applies_cli_crossing_weight_to_prior_tree():
    nodes = np.asarray(((0, 0), (1, 0), (2, 0), (0, 1), (2, 1)), dtype=float)
    adjacency = {index: [] for index in range(len(nodes))}
    _connect(adjacency, 0, 1, 1, "E")
    _connect(adjacency, 1, 2, 1, "E")
    _connect(adjacency, 0, 3, 1, "S")
    _connect(adjacency, 3, 4, 2, "E")
    _connect(adjacency, 4, 2, 1, "N")

    result = solve_line_graph_mst(
        nodes, adjacency, (0, 2), crossing_penalty=10,
        prior_segments=(((0.5, -1), (0.5, 0.5)),),
    )

    assert set(result.tree_edges) == {(0, 3), (3, 4), (2, 4)}
