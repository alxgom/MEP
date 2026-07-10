from shapely.geometry import GeometryCollection, Polygon

from vent_router.graphs import (
    add_bounds_axes,
    add_epsilon_axis_values,
    add_epsilon_geometry_axes,
    add_point_axes,
    add_polygon_vertex_axes,
    extend_allowed_boundary_axes,
    merge_close_values,
)


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


def test_add_epsilon_axis_values_adds_center_and_offsets():
    xs, ys = set(), set()

    add_epsilon_axis_values(xs, ys, (10.4, 20.6), epsilon=3)

    assert xs == {7, 10, 13}
    assert ys == {18, 21, 24}


def test_add_epsilon_geometry_axes_adds_vertices_holes_and_bounds():
    xs, ys = set(), set()
    poly = Polygon(
        [(0, 0), (20, 0), (20, 20), (0, 20)],
        holes=[[(5, 5), (10, 5), (10, 10), (5, 10)]],
    )

    add_epsilon_geometry_axes(xs, ys, poly, epsilon=2)

    assert {-2, 0, 2, 18, 20, 22}.issubset(xs)
    assert {3, 5, 7, 8, 10, 12}.issubset(ys)


def test_extend_allowed_boundary_axes_handles_empty_geometry():
    assert extend_allowed_boundary_axes(None) == ([], [])
    assert extend_allowed_boundary_axes(GeometryCollection()) == ([], [])


def test_extend_allowed_boundary_axes_returns_clustered_interior_axes():
    poly = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])

    xs, ys = extend_allowed_boundary_axes(poly, inset=100, cluster_dist=50)

    assert xs == [71, 929]
    assert ys == [71, 929]


def test_merge_close_values_deduplicates_and_thresholds_values():
    assert merge_close_values([0, 10, 15, 100], threshold=20) == [0, 100]


def test_merge_close_values_preserves_required_values():
    assert merge_close_values([0, 10, 15, 100], threshold=20, preserve_values={15}) == [0, 15, 100]


def test_merge_close_values_prefers_next_priority_value():
    assert merge_close_values([0, 30, 40, 100], threshold=20, priority_values={40}) == [0, 40, 100]


def test_merge_close_values_handles_empty_input():
    assert merge_close_values([], threshold=20) == []
