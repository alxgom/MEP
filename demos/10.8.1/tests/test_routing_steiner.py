from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mep_routing.routing import segments_from_edges, solve_steiner_tree


def adjacency(edges, *, reverse=False):
    result = {}
    for u, v, weight in reversed(edges) if reverse else edges:
        result.setdefault(u, []).append((v, weight, "axis"))
        result.setdefault(v, []).append((u, weight, "axis"))
    return result


def test_steiner_tree_connects_three_terminal_branch_and_is_immutable():
    nodes = np.array([(0, 0), (1, 0), (2, 0), (1, 1)], dtype=float)
    graph = adjacency([(0, 1, 1), (1, 2, 1), (1, 3, 1)])

    result = solve_steiner_tree(nodes, graph, [0, 2, 3], heuristic_directions=[(0, 0)])

    assert result.tree_nodes == frozenset({0, 1, 2, 3})
    assert result.tree_edges == ((0, 1), (1, 2), (1, 3))
    assert set(result.directed_edges) == {(0, 1), (1, 2), (1, 3)}
    assert segments_from_edges(nodes, result.tree_edges)[0] == ((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(FrozenInstanceError):
        result.tree_edges = ()


def test_disconnected_terminals_have_no_tree():
    nodes = np.array([(0, 0), (1, 0), (5, 0)], dtype=float)
    assert solve_steiner_tree(nodes, adjacency([(0, 1, 1)]), [0, 2]) is None


def test_duplicate_and_invalid_terminals_are_handled_explicitly():
    nodes = np.array([(0, 0), (1, 0)], dtype=float)
    graph = adjacency([(0, 1, 1)])

    assert solve_steiner_tree(nodes, graph, [0, 0]) is None
    result = solve_steiner_tree(nodes, graph, [0, 0, 1], heuristic_directions=[(0, 0)])
    assert result.tree_edges == ((0, 1),)
    with pytest.raises(ValueError, match="terminal node index"):
        solve_steiner_tree(nodes, graph, [0, 2])


def test_results_are_deterministic_and_solves_do_not_share_node_positions():
    nodes = np.array([(0, 0), (1, 0), (2, 0), (1, 1)], dtype=float)
    edges = [(0, 1, 1), (1, 2, 1), (1, 3, 1)]
    first = solve_steiner_tree(nodes, adjacency(edges), [0, 2, 3], heuristic_directions=[(0, 0)])
    other = solve_steiner_tree(np.array([(10, 10), (20, 10)]), adjacency([(0, 1, 10)]), [0, 1])
    repeated = solve_steiner_tree(nodes, adjacency(edges, reverse=True), [0, 2, 3], heuristic_directions=[(0, 0)])

    assert other.tree_edges == ((0, 1),)
    assert repeated == first


def test_min_edge_size_rejects_short_edges_including_two_terminal_tree():
    nodes = np.array([(0, 0), (1, 0)], dtype=float)
    graph = adjacency([(0, 1, 1)])

    assert solve_steiner_tree(nodes, graph, [0, 1], min_edge_size=1.0) is not None
    assert solve_steiner_tree(nodes, graph, [0, 1], min_edge_size=1.01) is None
