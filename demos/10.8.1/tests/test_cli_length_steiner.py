from itertools import combinations
import math
from types import SimpleNamespace

import numpy as np
import pytest

from mep_routing.installations.cli import directed_wall_graph
from mep_routing.installations.cli.length_steiner import (
    LengthSteinerStats, TreeLengthPolicy, segment_geometry,
    solve_length_steiner, validate_tree_lengths,
)


def prepared(points, edges, candidates, walls=None, weights=None):
    positions = np.asarray(points, dtype=float)
    graph = {index: [] for index in range(len(points))}
    for u, v in edges:
        weight = (weights or {}).get((u, v), math.dist(points[u], points[v]))
        graph[u].append((v, weight, directed_wall_graph._direction(points[u], points[v])))
        graph[v].append((u, weight, directed_wall_graph._direction(points[v], points[u])))
    return SimpleNamespace(
        nodes=positions, adjacency=graph, candidates=candidates,
        wall_points=walls or {(name, node): points[node]
                             for name, nodes in candidates.items() for node in nodes},
    )


@pytest.mark.parametrize("tees", (False, True))
def test_subdivided_straight_run_uses_geometry_not_edge_weight(tees):
    graph = prepared(((0, 0), (30, 0), (65, 0), (100, 0)),
                     ((0, 1), (1, 2), (2, 3)), {"a": [3]},
                     weights={(0, 1): 5000, (1, 2): 7000, (2, 3): 1000})
    result = solve_length_steiner(graph, 0, ("a",), 4000, TreeLengthPolicy(100, tees))
    assert result is not None
    assert result[0].score.total == 13000
    assert solve_length_steiner(graph, 0, ("a",), 4000, TreeLengthPolicy(100.1, tees)) is None


def test_machine_prefix_and_grille_adapter_extend_straight_runs():
    graph = prepared(((0, 0), (40, 0)), ((0, 1),), {"a": [1]},
                     walls={("a", 1): (70, 0)})
    policy = TreeLengthPolicy(100)
    result = solve_length_steiner(graph, 0, ("a",), 4000, policy,
                                  root_prefix=(((-30, 0), (0, 0)),))
    assert result is not None
    assert result[0].score.total == 70  # Prefix feasibility adds no new graph cost.
    assert solve_length_steiner(graph, 0, ("a",), 4000, policy) is None


def test_short_bend_and_short_terminal_piece_are_rejected():
    graph = prepared(((0, 0), (90, 0), (90, 200)), ((0, 1), (1, 2)), {"a": [2]})
    assert solve_length_steiner(graph, 0, ("a",), 4000, TreeLengthPolicy(100)) is None
    graph = prepared(((0, 0), (200, 0), (200, 90)), ((0, 1), (1, 2)), {"a": [2]})
    assert solve_length_steiner(graph, 0, ("a",), 4000, TreeLengthPolicy(100)) is None


def test_close_tees_are_valid_only_in_bends_mode():
    graph = prepared(((0, 0), (1000, 0), (1050, 0), (2000, 0),
                      (1000, 500), (1050, -500)),
                     ((0, 1), (1, 2), (2, 3), (1, 4), (2, 5)),
                     {"a": [4], "b": [5], "c": [3]})
    assert solve_length_steiner(graph, 0, ("a", "b", "c"), 4000, TreeLengthPolicy(100)) is not None
    assert solve_length_steiner(graph, 0, ("a", "b", "c"), 4000, TreeLengthPolicy(100, True)) is None


def test_strict_tree_can_choose_a_more_distant_grille_candidate():
    graph = prepared(((0, 0), (1000, 0), (1050, 0), (1400, 0), (2000, 0),
                      (1000, 500), (1050, -500), (1400, -500)),
                     ((0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (2, 6), (3, 7)),
                     {"a": [5], "b": [6, 7], "c": [4]})
    loose = solve_length_steiner(graph, 0, ("a", "b", "c"), 4000, TreeLengthPolicy(100))
    strict = solve_length_steiner(graph, 0, ("a", "b", "c"), 4000, TreeLengthPolicy(100, True))
    assert loose is not None and strict is not None
    assert loose[1]["b"] == 6
    assert strict[1]["b"] == 7
    assert strict[0].score.total >= loose[0].score.total


def test_tee_resets_straight_continuation_and_endpoint_length():
    graph = prepared(((0, 0), (200, 0), (250, 0), (200, 200)),
                     ((0, 1), (1, 2), (1, 3)), {"a": [2], "b": [3]})
    assert solve_length_steiner(graph, 0, ("a", "b"), 4000, TreeLengthPolicy(100)) is not None
    assert solve_length_steiner(graph, 0, ("a", "b"), 4000, TreeLengthPolicy(100, True)) is None


def test_root_without_prefix_can_fork_into_valid_endpoint_runs():
    graph = prepared(((0, 0), (200, 0), (0, 200)), ((0, 1), (0, 2)), {"a": [1], "b": [2]})
    assert solve_length_steiner(graph, 0, ("a", "b"), 4000, TreeLengthPolicy(100, True)) is not None


def test_three_outgoing_branches_share_one_real_fitting():
    graph = prepared(((0, 0), (200, 0), (400, 0), (200, 200), (200, -200)),
                     ((0, 1), (1, 2), (1, 3), (1, 4)), {"a": [2], "b": [3], "c": [4]})
    result = solve_length_steiner(graph, 0, ("a", "b", "c"), 17, TreeLengthPolicy(100, True))
    assert result is not None
    assert result[0].score.total == 834


def test_terminal_adapter_turn_and_diagonal_are_physical_constraints():
    graph = prepared(((0, 0), (200, 0)), ((0, 1),), {"a": [1]},
                     walls={("a", 1): (200, 99)})
    assert solve_length_steiner(graph, 0, ("a",), 4000, TreeLengthPolicy(100)) is None
    graph.wall_points[("a", 1)] = (200, 100)
    assert solve_length_steiner(graph, 0, ("a",), 4000, TreeLengthPolicy(100)) is not None
    graph.wall_points[("a", 1)] = (250, 50)
    assert solve_length_steiner(graph, 0, ("a",), 4000, TreeLengthPolicy(100)) is None


def test_physical_union_validator_rejects_cycles_disconnection_and_short_tees():
    root = (0, 0)
    policy = TreeLengthPolicy(100, True)
    valid = ((root, (200, 0)), ((200, 0), (400, 0)), ((200, 0), (200, 200)))
    ends = ((400, 0), (200, 200))
    assert validate_tree_lengths(valid, root, ends, policy)
    assert not validate_tree_lengths((*valid, ((400, 0), root)), root, ends, policy)
    assert not validate_tree_lengths((*valid, ((900, 900), (1200, 900))), root, ends, policy)
    assert not validate_tree_lengths(((root, (200, 0)), ((200, 0), (250, 0)),
                                      ((200, 0), (200, 200))), root, ((250, 0), (200, 200)), policy)


def test_saturation_and_reversal_do_not_create_length():
    policy = TreeLengthPolicy(100)
    run = policy.advance((None, 0), *segment_geometry((0, 0), (1000, 0)))
    assert run[1] == 100
    assert policy.advance(run, *segment_geometry((1000, 0), (0, 0))) is None
    assert policy.advance(run, *segment_geometry((1000, 0), (1000, 0))) == run


@pytest.mark.parametrize("minimum", (0, -1, math.inf, math.nan))
def test_invalid_threshold_is_rejected(minimum):
    with pytest.raises(ValueError):
        TreeLengthPolicy(minimum)


@pytest.mark.parametrize("tees", (False, True))
def test_small_graph_exhaustive_physical_tree_cost_oracle(tees):
    points = ((0, 0), (60, 0), (160, 0), (0, 120), (60, 120), (160, 120))
    edges = ((0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5))
    graph = prepared(points, edges, {"a": [2], "b": [5]})
    policy, bend = TreeLengthPolicy(100, tees), 17
    best = math.inf
    for count in range(1, len(edges) + 1):
        for chosen in combinations(edges, count):
            undirected = {}
            for u, v in chosen:
                undirected.setdefault(u, set()).add(v)
                undirected.setdefault(v, set()).add(u)
            parent, order, arcs = {0: None}, [0], []
            cyclic = False
            for u in order:
                for v in sorted(undirected.get(u, ())):
                    if v == parent[u]:
                        continue
                    if v in parent:
                        cyclic = True
                        break
                    parent[v] = u
                    order.append(v)
                    arcs.append((u, v))
            if cyclic or not {2, 5} <= parent.keys() or len(arcs) != len(chosen):
                continue
            if not validate_tree_lengths([(points[u], points[v]) for u, v in arcs], points[0],
                                         (points[2], points[5]), policy):
                continue
            value = 0
            for u, v in arcs:
                value += math.dist(points[u], points[v])
                if parent[u] is not None:
                    previous = directed_wall_graph._direction(points[parent[u]], points[u])
                    following = directed_wall_graph._direction(points[u], points[v])
                    value += bend if previous != following else 0
            best = min(best, value)
    stats = LengthSteinerStats()
    solved = solve_length_steiner(graph, 0, ("a", "b"), bend, policy, stats=stats)
    assert solved is not None
    assert solved[0].score.total == best
    assert stats.state_count > 0 and stats.subset_labels > 0


@pytest.mark.parametrize("tees", (False, True))
def test_compatible_minimum_preserves_baseline_weighted_objective(monkeypatch, tees):
    graph = prepared(((0, 0), (200, 0), (400, 0), (200, 200)),
                     ((0, 1), (1, 2), (1, 3)), {"a": [2], "b": [3]},
                     weights={(0, 1): 2000, (1, 2): 201, (1, 3): 202})
    monkeypatch.setattr(directed_wall_graph, "_prepare_wall_graph", lambda *args: graph)
    monkeypatch.setattr(directed_wall_graph, "_selected_connections", lambda *args: ())
    baseline = directed_wall_graph.solve_directed_wall_steiner(
        graph.nodes, graph.adjacency, 0, {}, (SimpleNamespace(route_name="a"), SimpleNamespace(route_name="b")), 4000,
    )
    hard = solve_length_steiner(graph, 0, ("a", "b"), 4000, TreeLengthPolicy(100, tees))
    assert hard is not None
    assert hard[0].score.total == baseline.tree.score.total == 6403
    assert hard[0].tree_edges == baseline.tree.tree_edges
