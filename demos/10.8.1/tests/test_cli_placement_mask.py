from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon, box

from mep_routing.installations.cli.placement_mask import CliPlacementMasks


def dwelling(rooms, walls=(), columns=(), shafts=(), allowed=None):
    return SimpleNamespace(
        rooms=[SimpleNamespace(polygon=room) for room in rooms],
        routing_region_base=box(0, 0, 1200, 800) if allowed is None else allowed,
        wall_polygons=walls, columns=columns, shafts=shafts,
    )


def test_placement_can_straddle_rooms_and_interior_walls():
    rooms = (box(0, 0, 600, 800), box(600, 0, 1200, 800))
    wall = box(550, 300, 650, 800)
    mask = CliPlacementMasks().get(dwelling(rooms, walls=(wall,)), (300, 100, 250), 0)
    assert mask.clear_room[8, 4] and mask.clear_room[8, 17]
    assert mask.visible[8, 11:13].all()
    assert mask.valid[8, 11:13].all() and not mask.clear_room[8, 11:13].any()
    # A wall-free opening is feasible too, but still isn't a clear single-room fit.
    assert mask.valid[3, 11] and not mask.clear_room[3, 11]
    assert not np.any(mask.valid & ~mask.visible)


def test_valid_tiles_fit_exact_allowed_region_and_clear_tiles_avoid_walls():
    room = Polygon([(0, 0), (1200, 0), (920, 800), (0, 800)])
    walls = (box(418, 100, 422, 700),)
    columns = (box(140, 200, 200, 270),)
    shafts = (box(760, 320, 820, 420),)
    source = dwelling((room,), walls, columns, shafts, allowed=room)
    mask = CliPlacementMasks().get(source, (200, 100, 250), 0)
    assert mask.valid.any()
    assert mask.valid[5, 8] and not mask.clear_room[5, 8]
    assert not mask.valid[4, 3] and not mask.valid[7, 15]
    for j, i in np.argwhere(mask.valid):
        x, y = (i + 0.5) * mask.step_mm, (j + 0.5) * mask.step_mm
        # Test the full region of possible centers colored green, not just its midpoint.
        half_tile = mask.step_mm / 2
        swept = box(x - 100 - half_tile, y - 50 - half_tile,
                    x + 100 + half_tile, y + 50 + half_tile)
        assert room.covers(swept)
        assert all(swept.disjoint(obstacle) for obstacle in (*columns, *shafts))
        if mask.clear_room[j, i]:
            assert all(swept.disjoint(wall) for wall in walls)


def test_allowed_space_outside_a_room_is_valid_but_an_excluded_wall_is_not():
    excluded = box(575, 0, 625, 800)
    allowed = box(0, 0, 1200, 800).difference(excluded)
    source = dwelling((box(0, 0, 600, 800),), walls=(excluded,), allowed=allowed)
    mask = CliPlacementMasks().get(source, (200, 100, 250), 0)
    assert mask.visible[6, 10] and not mask.valid[6, 10]
    assert mask.valid[6, 19] and not mask.clear_room[6, 19]


def test_rotation_dimension_and_dwelling_cache():
    room = box(0, 0, 500, 800)
    source = dwelling((room,), allowed=room)
    cache = CliPlacementMasks()
    horizontal = cache.get(source, (600, 200, 250), 0)
    vertical = cache.get(source, (600, 200, 250), 90)
    assert not horizontal.valid.any() and vertical.valid.any()
    assert cache.get(source, (600, 200, 250), 180) is horizontal
    assert cache.get(source, (600, 200, 250), 270) is vertical
    assert cache.get(source, (600, 200, 250), 45) is None
    assert not vertical.valid.flags.writeable
    assert not vertical.clear_room.flags.writeable
    assert cache.get(source, (150, 150, 250), 0).valid.any()
    smaller = box(0, 0, 100, 100)
    new_source = dwelling((smaller,), allowed=smaller)
    assert not cache.get(new_source, (600, 200, 250), 90).valid.any()
