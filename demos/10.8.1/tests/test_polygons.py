from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon

from vent_router.geometry import iter_polygons


def test_iter_polygons_yields_single_polygon():
    poly = Polygon([(0, 0), (1, 0), (0, 1)])

    assert list(iter_polygons(poly)) == [poly]


def test_iter_polygons_yields_polygons_from_multi_polygon():
    poly_a = Polygon([(0, 0), (1, 0), (0, 1)])
    poly_b = Polygon([(2, 2), (3, 2), (2, 3)])

    assert list(iter_polygons(MultiPolygon([poly_a, poly_b]))) == [poly_a, poly_b]


def test_iter_polygons_ignores_non_polygon_collection_items():
    poly = Polygon([(0, 0), (1, 0), (0, 1)])
    line = LineString([(0, 0), (1, 1)])

    assert list(iter_polygons(GeometryCollection([line, poly]))) == [poly]


def test_iter_polygons_handles_none_and_empty_geometry():
    assert list(iter_polygons(None)) == []
    assert list(iter_polygons(Polygon())) == []
