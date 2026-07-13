from shapely.geometry import box

import numpy as np

from vent_router.graphs import EnvView, build_axis_grid, build_regular_grid, filter_dynamic_machine_obstacle


def test_env_view_preserves_nodes_and_adjacency_references():
    nodes = [(0, 0), (100, 0)]
    adj = {0: [(1, 100.0, "R")], 1: [(0, 100.0, "L")]}

    env = EnvView(nodes, adj)

    assert env.nodes is nodes
    assert env.adj is adj


def test_env_view_uses_identity_equality_like_original_class():
    nodes = [(0, 0)]
    adj = {0: []}

    assert EnvView(nodes, adj) != EnvView(nodes, adj)


def test_regular_grid_keeps_interior_nodes_and_filters_wall_crossings():
    region = box(0, 0, 400, 400)

    nodes, edges = build_regular_grid(
        region,
        region,
        [box(150, 50, 250, 350)],
        grid_spacing_mm=100,
        wall_thickness_mm=20,
    )

    assert {tuple(node) for node in nodes} == {
        (100, 100), (200, 100), (300, 100),
        (100, 200), (200, 200), (300, 200),
        (100, 300), (200, 300), (300, 300),
    }
    assert len(edges) == 4
    assert {direction for _u, _v, _length, direction in edges} == {"N"}


def test_axis_grid_builds_visibility_filtered_edges_and_timing_breakdown():
    region = box(0, 0, 400, 400)

    nodes, edges, (nodes_ms, edges_ms) = build_axis_grid(
        [100, 200, 300],
        [100, 200, 300],
        region,
        region,
        [box(150, 50, 250, 350)],
        wall_thickness_mm=20,
    )

    assert len(nodes) == 9
    assert len(edges) == 4
    assert {direction for _u, _v, _length, direction in edges} == {"N"}
    assert nodes_ms >= 0
    assert edges_ms >= 0


def test_dynamic_machine_obstacle_preserves_protected_machine_access_nodes():
    nodes = np.array([(0, 0), (10, 0), (20, 0)], dtype=np.float32)
    edges = [(0, 1, 10.0, "E"), (1, 2, 10.0, "E")]
    edge_coords = np.array([(0, 0, 10, 0), (10, 0, 20, 0)], dtype=np.float32)
    machine = box(8, -2, 12, 2)

    blocked_env, blocked_nodes, blocked_edges = filter_dynamic_machine_obstacle(nodes, edges, edge_coords, machine, 0)
    protected_env, protected_nodes, protected_edges = filter_dynamic_machine_obstacle(nodes, edges, edge_coords, machine, 0, {1})

    assert blocked_nodes == 1
    assert blocked_edges == 2
    assert blocked_env.adj[0] == []
    assert protected_nodes == 0
    assert protected_edges == 0
    assert protected_env.adj[1] == [(0, 10.0, "W"), (2, 10.0, "E")]
