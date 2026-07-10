import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon

from vent_router.geometry import extract_boundary_segments, extract_line_segments


def test_extract_boundary_segments_from_polygon_exterior_and_hole():
    polygon = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 0)],
        holes=[[(2, 2), (4, 2), (2, 2)]],
    )

    segments = extract_boundary_segments(polygon)

    assert segments.shape == (6, 4)
    np.testing.assert_allclose(segments[0], [0, 0, 10, 0])
    np.testing.assert_allclose(segments[-2], [4, 2, 2, 2])


def test_extract_boundary_segments_filters_geometry_collection_to_polygons():
    collection = GeometryCollection([
        Polygon([(0, 0), (1, 0), (0, 0)]),
        LineString([(0, 0), (1, 1)]),
    ])

    segments = extract_boundary_segments(collection)

    assert segments.shape == (3, 4)


def test_extract_line_segments_handles_line_string_and_multi_line_string():
    line = LineString([(0, 0), (1, 0), (1, 1)])
    multi = MultiLineString([[(10, 10), (11, 10)], [(20, 20), (20, 21)]])

    np.testing.assert_allclose(
        extract_line_segments(line),
        [[0, 0, 1, 0], [1, 0, 1, 1]],
    )
    np.testing.assert_allclose(
        extract_line_segments(multi),
        [[10, 10, 11, 10], [20, 20, 20, 21]],
    )


def test_extract_line_segments_handles_none_and_empty_geometry():
    assert extract_line_segments(None).shape == (0, 4)
    assert extract_line_segments(LineString()).shape == (0, 4)
