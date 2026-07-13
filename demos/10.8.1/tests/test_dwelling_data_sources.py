from types import SimpleNamespace

from shapely.geometry import LineString, Polygon

from mep_routing.data_sources import (
    build_wall_polygons,
    choose_initial_machine_position,
    derive_room_boundary_walls,
)


def test_derive_room_boundary_walls_keeps_shared_boundaries_above_minimum_length():
    rooms = [
        SimpleNamespace(polygon=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])),
        SimpleNamespace(polygon=Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])),
    ]

    walls = derive_room_boundary_walls(rooms, columns=[], shafts=[])

    assert len(walls) == 1
    assert walls[0].equals(LineString([(100, 0), (100, 100)]))


def test_build_wall_polygons_excludes_structural_obstacles():
    walls = [LineString([(0, 0), (200, 0)])]
    column = Polygon([(80, -20), (120, -20), (120, 20), (80, 20)])

    polygons = build_wall_polygons(walls, columns=[column], shafts=[], wall_thickness_mm=20)

    assert len(polygons) == 1
    assert polygons[0].intersection(column).area == 0.0


def test_choose_initial_machine_position_prefers_wet_room_terminal_near_shaft():
    shaft = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    terminals = {
        "Kitchen": (5.0, 0.0),
        "Bathroom": (30.0, 0.0),
        "Toilet": (50.0, 0.0),
    }

    position = choose_initial_machine_position(
        terminals,
        shaft,
        lambda polygon: (polygon.representative_point().x, polygon.representative_point().y),
    )

    assert position == (30.0, 0.0)
