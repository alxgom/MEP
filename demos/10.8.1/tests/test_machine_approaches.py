"""Legal pin approaches share costs across baseline and length-aware search."""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from mep_routing.graphs.env import EnvView
from mep_routing.installations.sal import SAL_INSTALLATION, SalLiveRoutingSession, SalSolverSettings
from mep_routing.routing.metrics import count_route_short_pieces, count_ordered_route_turns
from mep_routing.routing.piece_search import PieceLengthPolicy
from mep_routing.routing.search import SearchBackend
from mep_routing.routing.segments import route_segments_from_path
from mep_routing.routing.weight_runtime import EdgeWeightOverlay, StaticClearanceCache


@pytest.fixture
def approaches():
    # The user's 90-degree pose: one straight lateral connection and one
    # longer connection with a bend. Extra edges exercise nearby traffic,
    # inward travel before the access point and an actual machine collision.
    nodes = np.array([
        (5400, 4994), (4997, 4994), (4850, 4994),
        (5400, 4744), (4750, 4744), (4750, 4894),
        (4850, 5004), (4997, 5004), (4840, 4994),
        (4700, 5250), (4650, 5250),
    ], dtype=float)
    pairs = ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (8, 2), (9, 10))
    adjacency = {i: [] for i in range(len(nodes))}
    edges = []
    for a, b in pairs:
        dx, dy = nodes[b] - nodes[a]
        direction = ("E" if dx > 0 else "W") if dy == 0 else ("N" if dy > 0 else "S")
        distance = math.dist(nodes[a], nodes[b])
        edges.append((a, b, distance, direction))
        adjacency[a].append((b, distance, direction))
        adjacency[b].append((a, distance, {"E": "W", "W": "E", "N": "S", "S": "N"}[direction]))
    env = EnvView(nodes, adjacency)
    workspace = SimpleNamespace(
        edge_list=edges, edge_coords=np.array([(*nodes[a], *nodes[b]) for a, b in pairs]),
        static_clearance_cache=StaticClearanceCache(), overlay=EdgeWeightOverlay(),
    )
    session = SalLiveRoutingSession(
        SAL_INSTALLATION, SAL_INSTALLATION.default_machine, workspace,
        None, (), (), (), (), (4561, 5250), 90,
        SalSolverSettings(4000, 5, 1.05, 0, 150, 1e9, 1.05, 1, 1),
        0, lambda *_args: 0,
    )
    return env, session


def test_legal_approaches_preserve_hard_blocks_and_unrelated_soft_costs(approaches):
    env, session = approaches
    runtime = session.routing_runtime(env)
    weights = {(1, 2): 900.0}  # A pre-existing cost from another layer survives.
    runtime.add_machine_clearance_weights(weights, 80)
    assert weights[(1, 2)] == 900
    assert (4, 5) not in weights
    assert weights[(6, 7)] > math.dist(env.nodes[6], env.nodes[7])
    assert weights[(2, 8)] > math.dist(env.nodes[2], env.nodes[8])
    assert weights[(9, 10)] == session.policy.overlap_block_weight

    # The exemption belongs to the connector's duct size, not any route
    # that happens to follow its axis.
    context = replace(runtime.context, machine_port_access_specs=tuple(
        dict(access, diameter_mm=90) for access in runtime.context.machine_port_access_specs
    ))
    from mep_routing.routing.weight_runtime import add_machine_clearance_weights
    mismatch = {}
    add_machine_clearance_weights(mismatch, 80, context)
    assert mismatch[(1, 2)] > math.dist(env.nodes[1], env.nodes[2])


def test_all_backends_receive_identical_weights_and_choose_valid_straight_access(approaches):
    env, session = approaches
    original_nodes = env.nodes.copy()
    original_adjacency = {node: list(edges) for node, edges in env.adj.items()}
    fields = []
    for backend in SearchBackend:
        runtime = replace(session, search_backend_index=int(backend)).routing_runtime(env)
        weights = {}
        runtime.add_route_clearance_weights(weights, "Washroom")
        # Identical preceding-route context for every backend.
        runtime.add_route_interaction_weights([((5100, 4500, 5500, 4500, "H"), 80)], 80, weights)
        fields.append(weights)
        targets = [dict(access, node_idx=2 if access["out_dir"] == "E" else 5)
                   for access in runtime.context.machine_port_access_specs if access["pin"] == "bl"]
        result = runtime.run_super_sink_search(
            [0, 3], ["bl"], {"bl": targets}, edge_weights=weights,
            piece_policy=PieceLengthPolicy(84, 168, "Washroom") if backend.enforces_min_pieces else None,
        )
        path, _, pin, target = result
        assert path == [0, 1, 2] and target["out_dir"] == "E"
        segments = route_segments_from_path("Washroom", path, env.nodes,
                                            pin_name=pin, global_pins={pin: target["pin_point"]}, target=target)
        assert count_ordered_route_turns("Washroom", segments) == 0
        assert count_route_short_pieces("Washroom", segments,
                                       lambda _name, terminal_segment=False: 84 if terminal_segment else 168) == 0
    assert all(field == fields[0] for field in fields)
    np.testing.assert_array_equal(env.nodes, original_nodes)
    assert env.adj == original_adjacency
