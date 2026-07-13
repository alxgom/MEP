from shapely.geometry import box

from vent_router.graphs import EnvView, build_regular_grid


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
