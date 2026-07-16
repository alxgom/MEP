from shapely.geometry import LineString, Polygon

from mep_routing.placement import candidate_machine_rooms, is_machine_placement_valid


class Room:
    def __init__(self, polygon=None, has_cover=False):
        if polygon is not None:
            self.polygon = polygon
        self.has_cover = has_cover


def _pins():
    return {
        "c_tl": (-1, 1),
        "c_tr": (1, 1),
        "c_br": (1, -1),
        "c_bl": (-1, -1),
    }


def test_candidate_machine_rooms_prefers_covered_rooms_with_enough_area():
    small = Room(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), has_cover=True)
    large_uncovered = Room(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), has_cover=False)
    large_covered = Room(Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]), has_cover=True)

    assert candidate_machine_rooms([small, large_uncovered, large_covered], min_area=100) == [large_covered]


def test_candidate_machine_rooms_falls_back_to_any_polygon_room():
    empty = Room(Polygon(), has_cover=True)
    valid = Room(Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]), has_cover=False)
    missing_polygon = Room()

    assert candidate_machine_rooms([empty, valid, missing_polygon], min_area=100) == [valid]


def test_candidate_machine_rooms_excludes_rooms_without_placeable_area():
    blocked = Room(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), has_cover=True)
    available = Room(Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]), has_cover=False)
    placeable_region = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])

    assert candidate_machine_rooms(
        [blocked, available], min_area=200, placeable_region=placeable_region,
    ) == [available]


def test_is_machine_placement_valid_checks_region_and_obstacles():
    region = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])

    assert is_machine_placement_valid(0, 0, _pins(), region, [], [], [])
    assert not is_machine_placement_valid(20, 20, _pins(), region, [], [], [])
    assert not is_machine_placement_valid(0, 0, _pins(), region, [LineString([(-2, 0), (2, 0)])], [], [])
    assert not is_machine_placement_valid(0, 0, _pins(), region, [], [Polygon([(-2, -2), (0, -2), (0, 0), (-2, 0)])], [])
    assert not is_machine_placement_valid(0, 0, _pins(), region, [], [], [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])])


def test_machine_footprint_must_be_fully_inside_routing_region():
    region = Polygon([(-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5)])
    partially_outside_pins = {
        "c_tl": (0, 1),
        "c_tr": (2, 1),
        "c_br": (2, -1),
        "c_bl": (0, -1),
    }

    assert not is_machine_placement_valid(
        1, 0, partially_outside_pins, region, (), (), (),
    )


def test_machine_footprint_must_fit_inside_one_room():
    region = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    left_room = Polygon([(-10, -10), (0, -10), (0, 10), (-10, 10)])
    right_room = Polygon([(0, -10), (10, -10), (10, 10), (0, 10)])
    left_room_pins = {
        "c_tl": (-6, 1),
        "c_tr": (-4, 1),
        "c_br": (-4, -1),
        "c_bl": (-6, -1),
    }

    assert is_machine_placement_valid(
        -5, 0, left_room_pins, region, (), (), (), room_regions=(left_room, right_room),
    )
    assert not is_machine_placement_valid(
        0, 0, _pins(), region, (), (), (), room_regions=(left_room, right_room),
    )


def test_machine_footprint_cannot_intersect_buffered_wall_geometry():
    region = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    wall_polygon = Polygon([(-2, -0.5), (2, -0.5), (2, 0.5), (-2, 0.5)])

    assert not is_machine_placement_valid(0, 0, _pins(), region, (wall_polygon,), (), ())
