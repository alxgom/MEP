import numpy as np

from mep_routing.routing import (
    add_machine_clearance_weights,
    add_route_interaction_weights,
    add_static_clearance_weights,
    block_terminal_node_edges,
    buffered_radius_mm,
    normalized_edge,
    required_clearance_mm,
    route_axis_records,
    set_block_weight,
    static_clearance_distances,
    weighted_edge_cost,
)


def test_buffered_radius_uses_ceiling_after_ratio():
    assert buffered_radius_mm(90, 1.05) == 48
    assert buffered_radius_mm(125, 1.05) == 66


def test_required_clearance_sums_buffered_radii():
    assert required_clearance_mm(90, 125, 1.05) == 114


def test_route_axis_records_keeps_axis_segments_with_route_diameter():
    records = route_axis_records(
        "Kitchen",
        [
            ((0, 0), (10, 0)),
            ((10, 0), (13, 4)),
            ((5, 5), (5, 10)),
        ],
        route_diameter=lambda route_name: 125 if route_name == "Kitchen" else 90,
    )

    assert records == [
        ((0.0, 0.0, 10.0, 0.0, "H"), 125),
        ((5.0, 5.0, 5.0, 10.0, "V"), 125),
    ]


def test_weighted_edge_cost_uses_normalized_edge_key_or_distance():
    weights = {(1, 3): 200.0}

    assert weighted_edge_cost(None, 3, 1, 50.0) == 50.0
    assert weighted_edge_cost(weights, 3, 1, 50.0) == 200.0
    assert weighted_edge_cost(weights, 2, 3, 50.0) == 50.0


def test_set_block_weight_normalizes_edge_and_returns_it():
    weights = {}

    edge = set_block_weight(weights, 5, 2, block_weight=1000.0)

    assert edge == (2, 5)
    assert weights == {(2, 5): 1000.0}
    assert normalized_edge(8, 3) == (3, 8)


def test_block_terminal_node_edges_skips_shaft_and_missing_nodes():
    weights = {}
    adj = {
        1: [(2, 10.0, "E"), (3, 20.0, "N")],
        4: [(5, 10.0, "E")],
    }

    blocked = block_terminal_node_edges(
        weights,
        adj,
        {"Shaft": 4, "Kitchen": 1, "Bath": 99},
        block_weight=5000.0,
    )

    assert blocked == [(1, 2), (1, 3)]
    assert weights == {(1, 2): 5000.0, (1, 3): 5000.0}


def test_static_clearance_distances_delegates_wall_and_shaft_distance_fields():
    edge_coords = np.array([[0.0, 0.0, 100.0, 0.0]])
    wall_segments = np.array([[0.0, 20.0, 100.0, 20.0]])
    shaft_segments = np.array([[50.0, -50.0, 50.0, 50.0]])

    wall_distances, shaft_distances = static_clearance_distances(
        edge_coords,
        wall_segments,
        shaft_segments,
    )

    assert wall_distances.tolist() == [20.0]
    assert shaft_distances.tolist() == [0.0]


def test_static_clearance_blocks_wall_and_shaft_edges_unless_entry_is_allowed():
    edge_list = [(0, 1, 0.0, "E"), (2, 3, 0.0, "E")]
    wall_distances = np.array([10.0, 100.0])
    shaft_distances = np.array([100.0, 10.0])

    weights = {}
    add_static_clearance_weights(
        weights,
        edge_list,
        wall_distances,
        shaft_distances,
        route_diameter_mm=100,
        buffer_ratio=1.0,
        shaft_clearance_mm=200,
        block_weight=1000.0,
    )
    assert weights == {(0, 1): 1000.0, (2, 3): 1000.0}

    entry_weights = {}
    add_static_clearance_weights(
        entry_weights,
        edge_list,
        wall_distances,
        shaft_distances,
        route_diameter_mm=100,
        buffer_ratio=1.0,
        shaft_clearance_mm=200,
        block_weight=1000.0,
        allow_shaft_entry=True,
    )
    assert entry_weights == {(0, 1): 1000.0}


def test_machine_clearance_blocks_envelope_edges_and_penalizes_soft_margin_edges():
    edge_list = [(0, 1, 0.0, "E"), (2, 3, 0.0, "E")]
    edge_coords = np.array([
        [-50.0, 0.0, 50.0, 0.0],
        [0.0, 100.0, 100.0, 100.0],
    ])
    nodes = np.array([[-50.0, 0.0], [50.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    weights = {}

    add_machine_clearance_weights(
        weights,
        edge_list,
        edge_coords,
        nodes,
        route_diameter_mm=50,
        buffer_ratio=1.0,
        machine_center=(0.0, 0.0),
        machine_angle_deg=0.0,
        machine_overall_width_mm=100,
        machine_body_height_mm=100,
        soft_margin_mm=50,
        clearance_penalty=100,
        block_weight=1000.0,
    )

    assert weights[(0, 1)] == 1000.0
    assert weights[(2, 3)] == 125.0


def test_route_interactions_block_overlaps_and_penalize_crossings_and_clearance():
    edge_list = [
        (0, 1, 0.0, "E"),
        (2, 3, 0.0, "N"),
        (4, 5, 0.0, "E"),
    ]
    edge_coords = np.array([
        [0.0, 0.0, 100.0, 0.0],
        [50.0, -50.0, 50.0, 50.0],
        [0.0, 30.0, 100.0, 30.0],
    ])
    nodes = np.array([
        [0.0, 0.0], [100.0, 0.0], [50.0, -50.0],
        [50.0, 50.0], [0.0, 30.0], [100.0, 30.0],
    ])
    weights = {}

    add_route_interaction_weights(
        [((0.0, 0.0, 100.0, 0.0, "H"), 40)],
        current_diameter_mm=40,
        accumulated_weights=weights,
        edge_list=edge_list,
        edge_coords=edge_coords,
        nodes=nodes,
        buffer_ratio=1.0,
        crossing_penalty=20.0,
        clearance_penalty=10.0,
        block_weight=1000.0,
    )

    assert weights == {(0, 1): 1000.0, (2, 3): 120.0, (4, 5): 110.0}
