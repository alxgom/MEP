"""Physical-piece feasibility and alternative-arrival solver contracts."""

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from mep_routing.routing.metrics import count_route_short_pieces
from mep_routing.routing.piece_search import PieceLengthPolicy, PieceSearchStats
from mep_routing.routing.search import SearchBackend, run_super_sink_search
from mep_routing.routing.segments import route_segments_from_path


BACKENDS = (SearchBackend.STATE_ASTAR_MIN_PIECES, SearchBackend.LINE_GRAPH_ASTAR_MIN_PIECES)


def graph(points, edges):
    adjacency = {index: [] for index in range(len(points))}
    for left, right in edges:
        a, b = points[left], points[right]
        direction = "E" if b[0] > a[0] else "W" if b[0] < a[0] else "N" if b[1] > a[1] else "S"
        reverse = {"E": "W", "W": "E", "N": "S", "S": "N"}[direction]
        length = math.dist(a, b)
        adjacency[left].append((right, length, direction))
        adjacency[right].append((left, length, reverse))
    return SimpleNamespace(nodes=np.array(points, dtype=float), adj=adjacency)


def solve(backend, env, starts, goal, *, endpoint=5, internal=15, target=None, **kwargs):
    target = target or {"pin": "p", "node_idx": goal, "in_dir": "E", "out_dir": "W"}
    return run_super_sink_search(
        backend, env, starts, ["p"], {"p": [target]}, 0,
        piece_policy=PieceLengthPolicy(endpoint, internal), heuristic_mode=1,
        **kwargs,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_short_internal_run_is_rejected_even_when_every_edge_is_traversable(backend):
    env = graph([(0, 0), (20, 0), (20, 10), (40, 10)], [(0, 1), (1, 2), (2, 3)])
    assert solve(backend, env, [0], 3)[0] is None
    assert run_super_sink_search(0, env, [0], ["p"], {
        "p": [{"pin": "p", "node_idx": 3, "in_dir": "E", "out_dir": "W"}],
    }, 0)[0] == [0, 1, 2, 3]


@pytest.mark.parametrize("backend", BACKENDS)
def test_collinear_edges_accumulate_and_endpoint_pieces_use_smaller_minimum(backend):
    env = graph([(0, 0), (5, 0), (5, 7), (5, 15), (10, 15)], [(0, 1), (1, 2), (2, 3), (3, 4)])
    result = solve(backend, env, [0], 4)
    assert result[0] == [0, 1, 2, 3, 4]
    assert result[1] == 25


@pytest.mark.parametrize("backend", BACKENDS)
def test_more_expensive_long_arrival_survives_cheaper_short_arrival(backend):
    # Both arrivals traverse 4->5 eastward. Only the route via 2 has enough
    # eastward length to turn north at 5. Its higher cost must not discard it.
    env = graph(
        [(0, 0), (20, 0), (0, 20), (20, 20), (25, 20), (30, 20), (30, 40)],
        [(0, 1), (1, 3), (0, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    )
    result = solve(backend, env, [0], 6, edge_weights={(0, 2): 25})
    assert result[0] == [0, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("backend", BACKENDS)
def test_connector_stub_extends_last_run_and_snap_bend_is_checked(backend):
    env = graph([(0, 0), (10, 0), (10, 20), (15, 20)], [(0, 1), (1, 2), (2, 3)])
    target = {"pin": "p", "node_idx": 3, "in_dir": "E", "out_dir": "W",
              "access_point": (15, 20), "pin_point": (25, 20)}
    path, _, pin, selected = solve(backend, env, [0], 3, endpoint=10, target=target)
    assert path is not None  # Last graph run is 5 mm; stub extends it to 15.
    segments = route_segments_from_path("Room", path, env.nodes, pin_name=pin,
                                        global_pins={pin: target["pin_point"]}, target=selected)
    assert count_route_short_pieces("Room", segments,
                                   lambda _name, terminal_segment=False: 10 if terminal_segment else 15) == 0
    # A snap adapter turns before the short 5 mm internal run is finished.
    target = dict(target, access_point=(15, 40), pin_point=(25, 40))
    assert solve(backend, env, [0], 3, endpoint=10, target=target)[0] is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_short_physical_stub_when_it_forms_its_own_endpoint_piece(backend):
    env = graph([(0, 0), (20, 0)], [(0, 1)])
    target = {"pin": "p", "node_idx": 1, "in_dir": "N", "out_dir": "S",
              "access_point": (20, 0), "pin_point": (20, 4)}
    assert solve(backend, env, [0], 1, target=target)[0] is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_source_adapter_can_extend_first_run_or_make_it_internal(backend):
    env = graph([(0, 0), (3, 0), (3, 20)], [(0, 1), (1, 2)])
    prefixes = {0: [((-10, 0), (0, 0))]}
    assert solve(backend, env, [0], 2, source_segments=prefixes)[0] == [0, 1, 2]
    prefixes = {0: [((0, -10), (0, 0))]}
    assert solve(backend, env, [0], 2, source_segments=prefixes)[0] is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_final_piece_can_finish_before_internal_minimum_and_sources_are_independent(backend):
    env = graph([(0, 0), (5, 0), (5, 10), (5, 3)], [(0, 1), (1, 3), (3, 2)])
    assert solve(backend, env, [0], 2)[0] == [0, 1, 3, 2]
    env = graph([(0, 0), (20, 0), (20, 20), (20, 17)], [(0, 1), (1, 3), (3, 2)])
    # The valid start must survive an invalid nearer start.
    assert solve(backend, env, [3, 0], 2, endpoint=8)[0] == [0, 1, 3, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_fractional_threshold_is_not_rounded_down(backend):
    env = graph([(0, 0), (94, 0)], [(0, 1)])
    assert solve(backend, env, [0], 1, endpoint=94.5, internal=189)[0] is None


def test_length_saturates_and_validation_rejects_invalid_policy_values():
    policy = PieceLengthPolicy(84, 168)
    run = policy.advance((None, 0, True), "E", 100)
    run = policy.advance(run, "E", 10000)
    assert run == ("E", 168, True)
    assert policy.advance(run, "N", 200) == ("N", 168, False)
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            PieceLengthPolicy(bad, 100)


@pytest.mark.parametrize("backend", BACKENDS)
def test_stats_measure_label_growth_without_changing_result(backend):
    env = graph([(0, 0), (10, 0), (20, 0)], [(0, 1), (1, 2)])
    stats = PieceSearchStats()
    assert solve(backend, env, [0], 2, stats=stats)[0] == [0, 1, 2]
    assert stats.expanded > 0 and stats.generated >= stats.expanded
    assert stats.peak_queue > 0 and stats.max_labels_per_state >= 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_cost_matches_exhaustive_materialized_routes_on_small_grids(backend):
    # Independent oracle: enumerate simple routes, then use the existing
    # physical-piece validator rather than the search's transition predicate.
    points = [(x, y) for y in (0, 7, 20) for x in (0, 7, 20)]
    edges = [(index, index + 1) for index in range(9) if index % 3 < 2]
    edges += [(index, index + 3) for index in range(6)]
    env = graph(points, edges)

    def paths(path):
        if path[-1] == 8:
            yield path
        else:
            for neighbour, _distance, _direction in env.adj[path[-1]]:
                if neighbour not in path:
                    yield from paths(path + [neighbour])

    feasible = []
    for path in paths([0]):
        segments = route_segments_from_path("Room", path, env.nodes)
        if count_route_short_pieces("Room", segments,
                                    lambda _name, terminal_segment=False: 5 if terminal_segment else 12) == 0:
            feasible.append(path)

    def cost(path, weights):
        result, incoming = 0, None
        for left, right in zip(path, path[1:]):
            distance, direction = next((d, direction) for n, d, direction in env.adj[left] if n == right)
            result += weights.get(tuple(sorted((left, right))), distance)
            result += 3 if incoming is not None and incoming != direction else 0
            incoming = direction
        return result + (3 if incoming != "E" else 0)

    rng = random.Random(181)
    for _ in range(12):
        weights = {tuple(sorted((a, b))): math.dist(points[a], points[b]) + rng.randrange(4) for a, b in edges}
        target = {"pin": "p", "node_idx": 8, "in_dir": "E", "out_dir": "W"}
        result = run_super_sink_search(
            backend, env, [0], ["p"], {"p": [target]}, 3,
            piece_policy=PieceLengthPolicy(5, 12), edge_weights=weights,
            heuristic_mode=1,
        )
        assert result[0] is not None
        assert cost(result[0], weights) == min(cost(path, weights) for path in feasible)
