import numpy as np

from mep_routing.graphs.runtime import (
    append_shaft_runtime_node,
    create_runtime_graph,
    restrict_pin_access_edges,
)


def test_runtime_graph_assembles_bidirectional_adjacency_and_spatial_index():
    runtime = create_runtime_graph(np.array([(0, 0), (100, 0)]), [(0, 1, 100.0, "E")])

    assert runtime.nodes.dtype == np.float32
    assert runtime.adjacency == {0: [(1, 100.0, "E")], 1: [(0, 100.0, "W")]}
    assert runtime.edge_coords.tolist() == [[0.0, 0.0, 100.0, 0.0]]
    assert runtime.spatial_index.query((101, 0))[1] == 1


def test_shaft_attachment_and_pin_access_filter_preserve_direction_contract():
    nodes, edges = append_shaft_runtime_node(
        np.array([(120, 50)]), [], shaft_center=(50, 50), shaft_bounds=(0, 0, 100, 100), clearance_mm=20,
    )
    filtered = restrict_pin_access_edges(
        [(0, 1, 70.0, "E"), (1, 2, 10.0, "E")], {1: "supply"}, {"supply": {"E"}},
    )

    assert nodes.tolist() == [[120, 50], [50, 50]]
    assert edges == [(0, 1, 70.0, "W")]
    assert filtered == [(1, 2, 10.0, "E")]
