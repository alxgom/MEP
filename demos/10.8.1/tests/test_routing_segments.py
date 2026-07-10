from vent_router.routing import (
    add_port_stub_segment,
    merged_axis_segments,
    merged_route_axis_segments,
    metric_route_segments,
    point_is_segment_endpoint,
)


def test_merged_axis_segments_merges_touching_collinear_pieces():
    route_segments = [
        ((0, 0), (10, 0)),
        ((10, 0), (20, 0)),
        ((5, 5), (5, 10)),
    ]

    assert sorted(merged_axis_segments(route_segments)) == [
        (0.0, 0.0, 20.0, 0.0, "H"),
        (5.0, 5.0, 5.0, 10.0, "V"),
    ]


def test_merged_route_axis_segments_keeps_route_names():
    routes = [
        ("Kitchen", [((0, 0), (10, 0)), ((10, 0), (20, 0))]),
        ("Bath", [((5, 5), (5, 10))]),
    ]

    assert sorted(merged_route_axis_segments(routes)) == [
        ("Bath", (5.0, 5.0, 5.0, 10.0, "V")),
        ("Kitchen", (0.0, 0.0, 20.0, 0.0, "H")),
    ]


def test_metric_route_segments_preserves_non_axis_segments():
    routes = [
        ("Kitchen", [((0, 0), (10, 0)), ((10, 0), (20, 0))]),
        ("Diagnostic", [((0, 0), (3, 4))]),
    ]

    assert sorted(metric_route_segments(routes)) == [
        ("Diagnostic", ((0.0, 0.0), (3.0, 4.0))),
        ("Kitchen", ((0.0, 0.0), (20.0, 0.0))),
    ]


def test_point_is_segment_endpoint_matches_either_endpoint():
    segment = ((0.0, 0.0), (10.0, 0.0))

    assert point_is_segment_endpoint((0.0, 0.0), segment)
    assert point_is_segment_endpoint((10.0, 0.0), segment)
    assert not point_is_segment_endpoint((5.0, 0.0), segment)


def test_add_port_stub_segment_adds_direct_segment_without_target_spec():
    segs = []
    nodes = [(0, 0)]
    global_pins = {"left_mid": (-10, 0)}

    add_port_stub_segment(segs, "left_mid", 0, global_pins, nodes)

    assert segs == [((0.0, 0.0), (-10.0, 0.0))]


def test_add_port_stub_segment_adds_access_bridge_when_needed():
    segs = []
    nodes = [(0, 0)]
    global_pins = {"left_mid": (-10, 0)}
    target_spec = {
        "access_point": (-5, 0),
        "pin_point": (-10, 0),
    }

    add_port_stub_segment(segs, "left_mid", 0, global_pins, nodes, target_spec)

    assert segs == [
        ((0.0, 0.0), (-5.0, 0.0)),
        ((-5.0, 0.0), (-10.0, 0.0)),
    ]


def test_add_port_stub_segment_ignores_missing_pin_or_node():
    segs = []

    add_port_stub_segment(segs, "missing", 0, {}, [(0, 0)])
    add_port_stub_segment(segs, "left_mid", None, {"left_mid": (0, 0)}, [(0, 0)])

    assert segs == []
