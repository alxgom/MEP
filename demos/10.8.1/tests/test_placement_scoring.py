import pytest
from shapely.geometry import Polygon

from vent_router.placement import (
    area_out_percentage,
    candidate_room_points,
    core_like_machine_candidate_score,
    machine_polygon_from_pins,
    point_angle_to_target,
    routing_frame_axes,
)


class Room:
    def __init__(self, polygon):
        self.polygon = polygon


def test_candidate_room_points_returns_centroid_and_axis_offsets_without_duplicates():
    room = Room(Polygon([(0, 0), (400, 0), (400, 400), (0, 400)]))

    assert candidate_room_points(room, axes=routing_frame_axes(), translation=100.0) == [
        (200, 200),
        (300, 200),
        (100, 200),
        (200, 300),
        (200, 100),
    ]


def test_machine_polygon_from_pins_and_area_out_percentage():
    pins = {
        "c_tl": (0, 10),
        "c_tr": (10, 10),
        "c_br": (10, 0),
        "c_bl": (0, 0),
    }
    room_poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])

    machine_poly = machine_polygon_from_pins(pins)

    assert machine_poly.area == 100.0
    assert area_out_percentage(machine_poly, room_poly) == 0.0
    assert area_out_percentage(machine_poly, Polygon([(5, 0), (20, 0), (20, 20), (5, 20)])) == 50.0


def test_point_angle_to_target_returns_signed_angle():
    assert point_angle_to_target((0, 0), (1.0, 0.0), (10, 0)) == 0.0
    assert point_angle_to_target((0, 0), (1.0, 0.0), (0, 10)) == 90.0
    assert point_angle_to_target((0, 0), (1.0, 0.0), (0, -10)) == -90.0


def test_core_like_machine_candidate_score_preserves_current_tuple_ordering():
    pins = {
        "left_mid": (-1, 0),
        "right_mid": (1, 0),
        "c_tl": (-1, 1),
        "c_tr": (1, 1),
        "c_br": (1, -1),
        "c_bl": (-1, -1),
    }
    room_poly = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
    boundary_distances = {(-1, 0): 5.0, (1, 0): 7.0}

    score = core_like_machine_candidate_score(
        cx=0,
        cy=0,
        angle=0,
        room_poly=room_poly,
        pins=pins,
        shaft_point=(-10, 0),
        kitchen_point=(10, 0),
        include_kitchen_distance=True,
        boundary_distance_fn=lambda point: boundary_distances[point],
        local_axis_to_world_fn=lambda local_vec, _angle: local_vec,
    )

    assert score == pytest.approx((0.0, 0.0, 0.0, 0.0, -5.0, -7.0, 13.5))
