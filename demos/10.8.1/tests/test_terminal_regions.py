import numpy as np
from shapely.geometry import LineString, Polygon, box

from mep_routing.routing import room_cover_geometry, terminal_boundary_segments, terminal_valid_region


def test_room_cover_geometry_and_valid_region_apply_cover_and_routing_constraints():
    room = box(0, 0, 10, 10)
    cover = box(2, 2, 12, 12)
    routing_region = box(0, 0, 8, 8)

    cover_geometry = room_cover_geometry(room, [cover])
    valid_region = terminal_valid_region(room, routing_region, cover_geometry)

    assert cover_geometry.equals(box(2, 2, 10, 10))
    assert valid_region.equals(box(2, 2, 8, 8))


def test_terminal_boundary_segments_combine_room_cover_wall_and_wall_polygon_constraints():
    room = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    segments = terminal_boundary_segments(
        room, [room], [LineString([(20, 0), (20, 10)])], [box(30, 0, 31, 1)], box(2, 2, 8, 8)
    )

    assert segments.shape == (17, 4)
    assert any(
        np.allclose(segment, [2, 2, 8, 2]) or np.allclose(segment, [8, 2, 2, 2])
        for segment in segments
    )
