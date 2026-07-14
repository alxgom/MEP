import numpy as np

from mep_routing.routing import (
    add_edge,
    build_pin_min_cost_flow_network,
    min_cost_flow,
    pin_target_specs,
    positive_flow_edges,
    small_pin_target_specs,
    source_start_nodes,
    trace_flow_path,
)


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


def test_source_start_nodes_returns_explicit_node_indices():
    assert source_start_nodes([1, 2, 3], kd=None) == [1, 2, 3]
    assert source_start_nodes([np.int64(4)], kd=None) == [4]
    assert source_start_nodes((), kd=None) == []


def test_source_start_nodes_queries_kd_for_coordinate_source():
    class FakeKd:
        def query(self, point):
            assert point == (10.0, 20.0)
            return 5.0, 7

    assert source_start_nodes((10.0, 20.0), FakeKd()) == [7]


def test_small_pin_target_specs_collects_targets_for_each_room():
    pin_node_map = {
        "tl": [{"pin": "tl", "node_idx": 1}],
        "tr": [{"pin": "tr", "node_idx": 2}],
        "left_mid": [{"pin": "left_mid", "node_idx": 99}],
    }

    specs = small_pin_target_specs(["Bath", "Toilet"], pin_node_map)

    assert specs == {
        "Bath": [{"pin": "tl", "node_idx": 1}, {"pin": "tr", "node_idx": 2}],
        "Toilet": [{"pin": "tl", "node_idx": 1}, {"pin": "tr", "node_idx": 2}],
    }
    assert pin_target_specs(["Supply"], pin_node_map, ("tr",)) == {
        "Supply": [{"pin": "tr", "node_idx": 2}],
    }


def test_pin_min_cost_flow_network_routes_start_to_matching_pin_target():
    target = {"pin": "left_mid", "node_idx": 1, "in_dir": "H"}
    network = build_pin_min_cost_flow_network(
        ["Shaft"],
        {"Shaft": [target]},
        {"Shaft": [0]},
        {0: [(1, 10.0, "H")], 1: [(0, 10.0, "H")]},
        edge_cost_fn=lambda _u, _v, distance: distance,
        direction_fn=lambda _u, _v: "H",
        bend_penalty=50.0,
        overlap_block_weight=1e9,
    )

    graph, source, sink, route_flow_nodes = network
    flow, cost = min_cost_flow(graph, source, sink, flow_required=1)
    path, selected_target = trace_flow_path(graph, route_flow_nodes["Shaft"], sink)

    assert flow == 1
    assert cost == 10.0
    assert path == [0, 1]
    assert selected_target == target


def test_pin_min_cost_flow_network_returns_none_without_targets():
    assert build_pin_min_cost_flow_network(
        ["Shaft"],
        {"Shaft": []},
        {"Shaft": [0]},
        {0: []},
        edge_cost_fn=lambda _u, _v, distance: distance,
        direction_fn=lambda _u, _v: "H",
        bend_penalty=0.0,
        overlap_block_weight=1e9,
    ) is None
