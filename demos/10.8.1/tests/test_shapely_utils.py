from shapely.geometry import GeometryCollection, LineString, Point, Polygon

from vent_router.geometry import snap_to_integer_grid


def test_snap_to_integer_grid_rounds_polygon_exterior_and_interiors():
    polygon = Polygon(
        [(0.2, 0.7), (10.4, 0.1), (10.6, 10.8), (0.2, 0.7)],
        holes=[[(2.2, 2.6), (4.4, 2.1), (2.2, 2.6)]],
    )

    snapped = snap_to_integer_grid(polygon)

    assert list(snapped.exterior.coords) == [(0.0, 1.0), (10.0, 0.0), (11.0, 11.0), (0.0, 1.0)]
    assert list(snapped.interiors[0].coords) == [(2.0, 3.0), (4.0, 2.0), (2.0, 3.0), (2.0, 3.0)]


def test_snap_to_integer_grid_rounds_line_string():
    line = LineString([(0.2, 0.7), (10.6, 10.2)])

    snapped = snap_to_integer_grid(line)

    assert list(snapped.coords) == [(0.0, 1.0), (11.0, 10.0)]


def test_snap_to_integer_grid_handles_geometry_collection():
    collection = GeometryCollection([
        LineString([(0.2, 0.7), (10.6, 10.2)]),
        Point(5.5, 6.5),
    ])

    snapped = snap_to_integer_grid(collection)

    assert snapped.geom_type in {"GeometryCollection", "LineString", "MultiLineString"}
    assert not snapped.is_empty
