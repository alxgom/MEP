from shapely.geometry import Polygon

from vent_router.graphs import add_bounds_axes, add_point_axes, add_polygon_vertex_axes, merge_close_values


def test_add_point_axes_rounds_point_coordinates_into_sets():
    xs, ys = set(), set()

    add_point_axes(xs, ys, (10.4, 20.6))

    assert xs == {10}
    assert ys == {21}


def test_add_polygon_vertex_axes_adds_exterior_and_interior_vertices():
    xs, ys = set(), set()
    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]],
    )

    add_polygon_vertex_axes(xs, ys, poly)

    assert xs == {0, 2, 4, 10}
    assert ys == {0, 2, 4, 10}


def test_add_bounds_axes_adds_buffered_bounds():
    xs, ys = set(), set()
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    add_bounds_axes(xs, ys, poly, clearance=5)

    assert xs == {-5, 15}
    assert ys == {-5, 15}


def test_merge_close_values_deduplicates_and_thresholds_values():
    assert merge_close_values([0, 10, 15, 100], threshold=20) == [0, 100]


def test_merge_close_values_preserves_required_values():
    assert merge_close_values([0, 10, 15, 100], threshold=20, preserve_values={15}) == [0, 15, 100]


def test_merge_close_values_prefers_next_priority_value():
    assert merge_close_values([0, 30, 40, 100], threshold=20, priority_values={40}) == [0, 40, 100]


def test_merge_close_values_handles_empty_input():
    assert merge_close_values([], threshold=20) == []
