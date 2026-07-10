from vent_router.routing import add_edge, min_cost_flow, positive_flow_edges, trace_flow_path


def test_min_cost_flow_chooses_lower_cost_path():
    graph = [[] for _ in range(4)]
    add_edge(graph, 0, 1, 1, 1)
    add_edge(graph, 1, 3, 1, 1)
    add_edge(graph, 0, 2, 1, 5)
    add_edge(graph, 2, 3, 1, 1)

    flow, cost = min_cost_flow(graph, source=0, sink=3, flow_required=1)

    assert flow == 1
    assert cost == 2.0
    assert [edge["to"] for edge in positive_flow_edges(graph, 0)] == [1]


def test_min_cost_flow_returns_partial_flow_when_sink_capacity_is_limited():
    graph = [[] for _ in range(3)]
    add_edge(graph, 0, 1, 2, 1)
    add_edge(graph, 1, 2, 1, 1)

    flow, cost = min_cost_flow(graph, source=0, sink=2, flow_required=2)

    assert flow == 1
    assert cost == 2.0


def test_trace_flow_path_reconstructs_state_edges_and_target():
    graph = [[] for _ in range(5)]
    target = {"pin": "s1"}
    add_edge(graph, 0, 1, 1, 0, ("state", 10, 20))
    add_edge(graph, 1, 2, 1, 0, ("state", 20, 30))
    add_edge(graph, 2, 3, 1, 0, ("target", target))
    add_edge(graph, 3, 4, 1, 0)
    min_cost_flow(graph, source=0, sink=4, flow_required=1)

    path, traced_target = trace_flow_path(graph, 0, 4)

    assert path == [10, 20, 30]
    assert traced_target == target
    assert positive_flow_edges(graph, 0) == []


def test_trace_flow_path_returns_none_when_path_is_incomplete():
    graph = [[] for _ in range(2)]
    add_edge(graph, 0, 1, 1, 0, ("state", 10, 20))
    min_cost_flow(graph, source=0, sink=1, flow_required=1)

    assert trace_flow_path(graph, 0, 1) == (None, None)
