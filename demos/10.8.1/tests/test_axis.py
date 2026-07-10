from vent_router.geometry import (
    axis_segment_distance,
    axis_segment_relation,
    normalize_axis_segment,
)


def test_normalize_axis_segment_orders_horizontal_and_vertical_segments():
    assert normalize_axis_segment((10, 5), (0, 5)) == (0.0, 5.0, 10.0, 5.0, "H")
    assert normalize_axis_segment((3, 9), (3, 1)) == (3.0, 1.0, 3.0, 9.0, "V")


def test_normalize_axis_segment_rejects_zero_length_and_diagonal_segments():
    assert normalize_axis_segment((1, 1), (1, 1)) is None
    assert normalize_axis_segment((0, 0), (1, 1)) is None


def test_axis_segment_relation_detects_overlap_and_crossing():
    horizontal = normalize_axis_segment((0, 0), (10, 0))
    overlapping = normalize_axis_segment((5, 0), (15, 0))
    crossing = normalize_axis_segment((7, -5), (7, 5))

    assert axis_segment_relation(horizontal, overlapping) == "overlap"
    assert axis_segment_relation(horizontal, crossing) == "cross"


def test_axis_segment_relation_ignores_touching_endpoint_as_overlap():
    left = normalize_axis_segment((0, 0), (10, 0))
    right = normalize_axis_segment((10, 0), (20, 0))

    assert axis_segment_relation(left, right) is None


def test_axis_segment_distance_uses_axis_aligned_bounding_gap():
    horizontal = normalize_axis_segment((0, 0), (10, 0))
    separated = normalize_axis_segment((15, 12), (20, 12))

    assert axis_segment_distance(horizontal, separated) == 13.0

