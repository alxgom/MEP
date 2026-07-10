from vent_router.graphs import EnvView


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
