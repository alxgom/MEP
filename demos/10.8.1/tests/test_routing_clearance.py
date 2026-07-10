from vent_router.routing import (
    buffered_radius_mm,
    required_clearance_mm,
    route_axis_records,
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
