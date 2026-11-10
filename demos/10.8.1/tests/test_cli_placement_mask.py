from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import numpy as np
from shapely.geometry import Polygon, box

from mep_routing.installations.cli.catalog import CLI_PEAD_MACHINES, CliConnector
from mep_routing.installations.cli.placement_mask import (
    CliPlacementMask, CliPlacementMasks, cell_center_product_candidates,
)


def machine(width=200, depth=100, outlet_width=100):
    return replace(
        CLI_PEAD_MACHINES[0], body_size_mm=(width, depth, 250),
        connectors=(CliConnector("air_out", "Supply", (0, depth // 2, 0), (0, 1), (outlet_width, 178)),),
    )


def dwelling(rooms, walls=(), columns=(), shafts=(), allowed=None):
    return SimpleNamespace(
        rooms=[SimpleNamespace(polygon=room) for room in rooms],
        routing_region_base=box(0, 0, 1200, 1400) if allowed is None else allowed,
        wall_polygons=walls, columns=columns, shafts=shafts,
    )


def test_placement_can_straddle_rooms_and_interior_walls():
    rooms = (box(0, 0, 600, 1400), box(600, 0, 1200, 1400))
    wall = box(550, 300, 650, 1400)
    mask = CliPlacementMasks().get(dwelling(rooms, walls=(wall,)), machine(300), 0)
    assert mask.clear_room[8, 4] and mask.clear_room[8, 17]
    assert mask.visible[8, 11:13].all()
    assert mask.valid[8, 11:13].all() and not mask.clear_room[8, 11:13].any()
    # A wall-free opening is feasible too, but still isn't a clear single-room fit.
    open_mask = CliPlacementMasks().get(dwelling(rooms), machine(300), 0)
    assert open_mask.valid[3, 11] and not open_mask.clear_room[3, 11]
    assert not np.any(mask.valid & ~mask.visible)


def test_valid_tiles_fit_exact_allowed_region_and_clear_tiles_avoid_walls():
    room = Polygon([(0, 0), (1200, 0), (920, 1400), (0, 1400)])
    walls = (box(418, 100, 422, 700),)
    columns = (box(140, 200, 200, 270),)
    shafts = (box(760, 320, 820, 420),)
    source = dwelling((room,), walls, columns, shafts, allowed=room)
    mask = CliPlacementMasks().get(source, machine(), 0)
    assert mask.valid.any()
    assert mask.valid[5, 8] and not mask.clear_room[5, 8]
    assert not mask.valid[4, 3] and not mask.valid[7, 15]
    for j, i in np.argwhere(mask.valid):
        x, y = (i + 0.5) * mask.step_mm, (j + 0.5) * mask.step_mm
        # Test both swept rectangles over the entire center tile.
        half_tile = mask.step_mm / 2
        body = box(x - 100 - half_tile, y - 50 - half_tile,
                   x + 100 + half_tile, y + 50 + half_tile)
        extension = box(x - 50 - half_tile, y + 50 - half_tile,
                        x + 50 + half_tile, y + 550 + half_tile)
        for swept in (body, extension):
            assert room.covers(swept)
            assert all(swept.disjoint(obstacle) for obstacle in (*columns, *shafts))
        if mask.clear_room[j, i]:
            assert all(body.disjoint(wall) for wall in walls)


def test_allowed_space_outside_a_room_is_valid_but_an_excluded_wall_is_not():
    excluded = box(575, 0, 625, 800)
    allowed = box(0, 0, 1200, 800).difference(excluded)
    source = dwelling((box(0, 0, 600, 800),), walls=(excluded,), allowed=allowed)
    mask = CliPlacementMasks().get(source, machine(), 0)
    assert mask.visible[3, 10] and not mask.valid[3, 10]
    assert mask.valid[3, 19] and not mask.clear_room[3, 19]


def test_rotation_dimension_and_dwelling_cache():
    room = box(0, 0, 900, 1100)
    source = dwelling((room,), allowed=room)
    cache = CliPlacementMasks()
    large = machine(1000, 200)
    horizontal = cache.get(source, large, 0)
    vertical = cache.get(source, large, 90)
    assert not horizontal.valid.any() and vertical.valid.any()
    assert cache.get(source, large, 360) is horizontal
    assert cache.get(source, large, 450) is vertical
    assert cache.get(source, large, 45) is None
    assert not vertical.valid.flags.writeable
    assert not vertical.clear_room.flags.writeable
    assert cache.get(source, machine(150, 150), 0).valid.any()
    smaller = box(0, 0, 100, 100)
    new_source = dwelling((smaller,), allowed=smaller)
    assert not cache.get(new_source, large, 90).valid.any()


def test_supply_extension_blocks_poses_at_the_boundary_in_its_own_direction():
    room = box(0, 0, 1200, 1400)
    source = dwelling((room,), allowed=room)
    cache = CliPlacementMasks()
    north = cache.get(source, machine(), 0)
    south = cache.get(source, machine(), 180)
    assert north.valid[3, 8] and not south.valid[3, 8]
    assert south.valid[24, 8] and not north.valid[24, 8]


def test_extension_controls_fit_but_only_the_body_controls_room_color():
    room = box(0, 0, 1200, 1400)
    cache = CliPlacementMasks()
    # At center (425, 175) the narrow extension reaches the middle obstacle,
    # but misses the column beside it and the body is below both.
    beside = box(510, 460, 525, 490)
    ahead = box(400, 460, 450, 490)
    source = dwelling((room,), columns=(beside,))
    assert cache.get(source, machine(), 0).clear_room[3, 8]
    assert not cache.get(source, machine(outlet_width=200), 0).valid[3, 8]
    blocked = cache.get(dwelling((room,), shafts=(ahead,)), machine(), 0)
    assert not blocked.valid[3, 8] and not blocked.clear_room[3, 8]
    # The body is clear in the lower room; the extension crosses into the upper one.
    rooms = (box(0, 0, 1200, 450), box(0, 450, 1200, 1400))
    mask = cache.get(dwelling(rooms, walls=(ahead,)), machine(), 0)
    assert mask.valid[3, 8] and mask.clear_room[3, 8]


def test_cell_center_product_adds_only_valid_coordinate_pairs():
    cells = ((0, 0, 50, 50), (50, 50, 100, 100), (0, 100, 50, 150))
    valid = np.array([[True, True], [False, False], [False, True]])
    clear = np.array([[True, False], [False, False], [False, True]])
    mask = CliPlacementMask((0, 0), 50, np.ones_like(valid), valid, clear)

    assert cell_center_product_candidates(mask, cells) == (
        ((25, 25), "clear"),
        ((75, 25), "allowed"),
        ((75, 125), "clear"),
    )
