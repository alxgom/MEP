from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon, box

from mep_routing.installations.cli.placement_mask import CliPlacementMasks


def dwelling(rooms, walls=(), columns=(), shafts=()):
    return SimpleNamespace(
        rooms=[SimpleNamespace(polygon=room) for room in rooms],
        routing_region_base=box(0, 0, 1200, 800),
        wall_polygons=walls, columns=columns, shafts=shafts,
    )


def test_no_placement_may_straddle_touching_rooms():
    rooms = (box(0, 0, 600, 800), box(600, 0, 1200, 800))
    mask = CliPlacementMasks().get(dwelling(rooms), (300, 100, 250), 0)
    assert mask.valid[6, 4] and mask.valid[6, 16]
    assert mask.visible[6, 11:13].all()
    assert not mask.valid[6, 11:13].any()
    assert not np.any(mask.valid & ~mask.visible)


def test_valid_tiles_fit_exact_geometry_including_sloped_room_and_thin_obstacles():
    room = Polygon([(0, 0), (1200, 0), (920, 800), (0, 800)])
    walls = (box(418, 100, 422, 700),)
    columns = (box(140, 200, 200, 270),)
    shafts = (box(760, 320, 820, 420),)
    source = dwelling((room,), walls, columns, shafts)
    mask = CliPlacementMasks().get(source, (200, 100, 250), 0)
    assert mask.valid.any()
    assert mask.visible[5, 8] and not mask.valid[5, 8]
    for j, i in np.argwhere(mask.valid):
        x, y = (i + 0.5) * mask.step_mm, (j + 0.5) * mask.step_mm
        # Test the full region of possible centers colored green, not just its midpoint.
        half_tile = mask.step_mm / 2
        swept = box(x - 100 - half_tile, y - 50 - half_tile,
                    x + 100 + half_tile, y + 50 + half_tile)
        assert room.covers(swept)
        assert all(swept.disjoint(obstacle) for obstacle in (*walls, *columns, *shafts))


def test_rotation_dimension_and_dwelling_cache():
    source = dwelling((box(0, 0, 500, 800),))
    cache = CliPlacementMasks()
    horizontal = cache.get(source, (600, 200, 250), 0)
    vertical = cache.get(source, (600, 200, 250), 90)
    assert not horizontal.valid.any() and vertical.valid.any()
    assert cache.get(source, (600, 200, 250), 180) is horizontal
    assert cache.get(source, (600, 200, 250), 270) is vertical
    assert cache.get(source, (600, 200, 250), 45) is None
    assert not vertical.valid.flags.writeable
    assert cache.get(source, (150, 150, 250), 0).valid.any()
    new_source = dwelling((box(0, 0, 100, 100),))
    assert not cache.get(new_source, (600, 200, 250), 90).valid.any()
